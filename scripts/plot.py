#!/usr/bin/env python3
"""Plot and tabulate halo3d results from the CSV rows written by --csv (stdlib csv + matplotlib only).

Every configuration is run several times; the plots and tables use the MEDIAN time and show the spread
(min..max over the repetitions) as error bars.

  python3 scripts/plot.py --strong results/cpu_strong.csv results/gpu_strong.csv \
      --weak results/cpu_weak.csv results/gpu_weak.csv \
      --threads results/cpu_threads.csv --stream results/cpu_bw.csv \
      --roof "CPU node (2 sockets)|results/cpu_hybrid.csv|2|<copy GB/s>|<spec GB/s>|<FP64 GFLOP/s>" \
      --roof "A100 80GB PCIe|results/gpu_strong.csv|1|<copy GB/s>|<spec GB/s>|<FP64 GFLOP/s>" \
      --table --out results

Writes scaling.png (strong speedup, strong and weak efficiency), breakdown.png (per-step time: interior,
halo wait, rest), threads.png (single-node thread scaling against the measured bandwidth ceiling) and
roofline.png (one panel per --roof, measured copy bandwidth as the roof, spec sheet dashed).
"""
import argparse
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOP_PER_LUP, BYTE_PER_LUP = 8, 16  # 6 adds + 2 multiplies; read u once + write v once (FP64)
AI = FLOP_PER_LUP / BYTE_PER_LUP
# fixed categorical order (validated palette), plus a marker per series so colour never carries identity alone
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = "osD^vP*X"
IDEAL = "#8a8a8a"
INK = "#444444"


def label_of(r):
    if r["backend"] == "omp":
        return f"CPU, {r['threads']} threads/rank"
    return "GPU, CUDA-aware MPI" if r["cuda_aware"] == "1" else "GPU, host-staged"


def load(paths, per_rank=False):
    """{label: [point, ...]} sorted by ranks; point = dict with median/min/max time and medians of the rest.
    per_rank=True (weak scaling) adds the block size per rank to the label."""
    groups = {}
    for path in paths or []:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                label = label_of(r)
                if per_rank:
                    label += f", {int(r['nx']) // int(r['px'])}^3 per rank"
                key = (label, int(r["ranks"]), r["nx"], r["ny"], r["nz"])
                groups.setdefault(key, []).append(r)
    series = {}
    for (label, ranks, nx, ny, nz), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        t = [float(r["time_s"]) for r in rows]
        steps = int(rows[0]["steps"])
        lups = int(nx) * int(ny) * int(nz) * steps
        med = st.median(t)
        series.setdefault(label, []).append({
            "ranks": ranks, "nodes": int(rows[0].get("nodes", 0) or 0), "threads": int(rows[0]["threads"]),
            "grid": f"{nx}x{ny}x{nz}", "steps": steps, "n": len(t), "t": med, "tmin": min(t), "tmax": max(t),
            "wait": st.median(float(r["wait_s"]) for r in rows),
            "inner": st.median(float(r.get("inner_s", 0) or 0) for r in rows),
            "glups": lups / med / 1e9,
            "kernel_gbs": st.median(float(r["kernel_gbs"]) for r in rows),
            "err": max(float(r["err"]) for r in rows),
        })
    return series


def style(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, which="major", lw=0.5, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def efficiency(rows, kind):
    """(ranks, value, lower, upper) per point; strong anchors at the smallest rank count, weak at its time."""
    p0, t0 = rows[0]["ranks"], rows[0]["t"]
    out = []
    for r in rows:
        f = {"speedup": lambda t: p0 * t0 / t,
             "strong": lambda t: p0 * t0 / t / r["ranks"],
             "weak": lambda t: t0 / t}[kind]
        out.append((r["ranks"], f(r["t"]), f(r["tmax"]), f(r["tmin"])))
    return out


def scaling(strong, weak, look, path):
    panels = []
    if strong:
        panels += [("speedup", strong), ("strong", strong)]
    if weak:
        panels.append(("weak", weak))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2), squeeze=False)
    for ax, (kind, data) in zip(axes[0], panels):
        pmax = 1
        for label, rows in data.items():
            pts = efficiency(rows, kind)
            ps = [p[0] for p in pts]
            y = [p[1] for p in pts]
            err = [[max(0.0, p[1] - p[2]) for p in pts], [max(0.0, p[3] - p[1]) for p in pts]]
            ax.errorbar(ps, y, yerr=err, lw=2, ms=7, capsize=3, label=label, **look[label])
            pmax = max(pmax, ps[-1])
        if kind == "speedup":
            ax.plot([1, pmax], [1, pmax], ls="--", lw=1, color=IDEAL, label="ideal")
            ax.set_yscale("log", base=2)
            ax.set_yticks([2 ** e for e in range(0, pmax.bit_length())], [str(2 ** e) for e in range(0, pmax.bit_length())])
            style(ax, "ranks (1 rank = 1 CPU socket or 1 GPU)", "speedup (1 CPU node counted as 2 ranks)",
                  "Strong scaling: speedup")
        else:
            ax.axhline(1.0, ls="--", lw=1, color=IDEAL, label="ideal")
            ax.set_ylim(0, 1.15)
            style(ax, "ranks (1 rank = 1 CPU socket or 1 GPU)", "parallel efficiency",
                  "Strong scaling: efficiency" if kind == "strong" else "Weak scaling: efficiency")
        ax.set_xscale("log", base=2)
        ticks = [2 ** e for e in range(0, pmax.bit_length())]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def breakdown(data, path, title):
    """Small multiples, one per series: whole step, interior update and time blocked in MPI_Waitall, side by side.
    Not stacked: on the GPU the host waits in MPI while the interior kernel runs, so the parts overlap in time.
    step - interior is the time the overlap did not hide."""
    labels = list(data)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.0), squeeze=False)
    parts = [("whole step", COLORS[0]), ("interior update", COLORS[1]), ("MPI_Waitall", COLORS[3])]
    w = 0.27
    for ax, label in zip(axes[0], labels):
        rows = data[label]
        xs = list(range(len(rows)))
        vals = [[1e3 * r[key] / r["steps"] for r in rows] for key in ("t", "inner", "wait")]
        for k, ((name, color), v) in enumerate(zip(parts, vals)):
            ax.bar([x + (k - 1) * w for x in xs], v, w, color=color, edgecolor="white", linewidth=1.5, label=name)
        ax.set_xticks(xs, [str(r["ranks"]) for r in rows])
        style(ax, "ranks", "ms per step (median run)", label)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def threads_plot(series, stream, path):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    rows = sorted((r for rows in series.values() for r in rows), key=lambda r: r["threads"])
    ts = [r["threads"] for r in rows]
    y = [r["glups"] for r in rows]
    err = [[y_ - r["glups"] * r["t"] / r["tmax"] for y_, r in zip(y, rows)],
           [r["glups"] * r["t"] / r["tmin"] - y_ for y_, r in zip(y, rows)]]
    ax.errorbar(ts, y, yerr=err, lw=2, ms=7, capsize=3, color=COLORS[0], marker="o", label="stencil, 1 rank, 512^3")
    if stream:
        s = sorted(stream, key=lambda r: r[0])
        ax.plot([r[0] for r in s], [r[1] / BYTE_PER_LUP for r in s], lw=2, ms=7, color=COLORS[1], marker="s",
                ls="--", label="ceiling: measured copy GB/s / 16 B")
    ax.axvline(24, ls=":", lw=1, color=IDEAL)
    ax.text(24.5, ax.get_ylim()[1] * 0.05, "socket 0 full", fontsize=7, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ts, [str(t) for t in ts])
    style(ax, "OpenMP threads (packed: socket 0 first)", "GLUP/s", "One CPU node: thread scaling")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def roofline(roofs, path):
    fig, axes = plt.subplots(1, len(roofs), figsize=(5.6 * len(roofs), 4.4), squeeze=False)
    ai = [2 ** (e / 4) for e in range(-16, 25)]  # 1/16 .. 64 flop/byte
    for ax, (name, csv_path, ranks, bw, spec_bw, fp64) in zip(axes[0], roofs):
        ax.plot(ai, [min(bw * x, fp64) for x in ai], lw=2, color=INK, label=f"measured roof: {bw:g} GB/s copy")
        ax.plot(ai, [min(spec_bw * x, fp64) for x in ai], lw=1, ls="--", color=IDEAL,
                label=f"spec sheet: {spec_bw:g} GB/s, {fp64:g} GFLOP/s")
        ax.axvline(AI, ls=":", lw=0.8, color=IDEAL)
        series = load([csv_path])
        k = 0
        for label, rows in series.items():
            pick = [r for r in rows if r["ranks"] == ranks]
            if not pick or "CUDA-aware" in label:   # on one device there is no halo: same numbers as staged
                continue
            r = max(pick, key=lambda r: r["glups"])
            ax.plot(AI, FLOP_PER_LUP * r["glups"], ls="none", ms=9, color=COLORS[k], marker=MARKERS[k],
                    label=f"{label}: end-to-end {r['grid']}, {FLOP_PER_LUP * r['glups']:.0f} GFLOP/s")
            if r["kernel_gbs"] > 0 and abs(r["kernel_gbs"] * AI / (FLOP_PER_LUP * r["glups"]) - 1) > 0.02:
                ax.plot(AI, r["kernel_gbs"] * AI, ls="none", ms=9, mfc="none", color=COLORS[k], marker=MARKERS[k],
                        label=f"{label}: interior kernel, {r['kernel_gbs'] * AI:.0f} GFLOP/s")
            k += 1
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, "arithmetic intensity (flop/byte)", "FP64 GFLOP/s", f"Roofline: {name}")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def table(title, data, kind):
    print(f"\n### {title}\n")
    print("| series | nodes | ranks | global grid | runs | ms/step median (min-max) | GLUP/s | "
          f"{'speedup vs first row | ' if kind == 'strong' else ''}efficiency | wait ms/step | interior ms/step | max rel. error |")
    print("|---|---|---|---|---|---|---|" + ("---|" if kind == "strong" else "") + "---|---|---|---|")
    for label, rows in data.items():
        eff = efficiency(rows, kind)
        spd = efficiency(rows, "speedup")
        for r, e, s in zip(rows, eff, spd):
            ms = lambda x: 1e3 * x / r["steps"]
            print(f"| {label} | {r['nodes']} | {r['ranks']} | {r['grid']} | {r['n']} | {ms(r['t']):.2f} "
                  f"({ms(r['tmin']):.2f}-{ms(r['tmax']):.2f}) | {r['glups']:.2f} | "
                  + (f"{rows[0]['t'] / r['t']:.2f} | " if kind == "strong" else "")
                  + f"{e[1]:.2f} | {ms(r['wait']):.2f} | {ms(r['inner']):.2f} | {r['err']:.1e} |")


def thread_table(series, stream):
    rows = sorted((r for rows in series.values() for r in rows), key=lambda r: r["threads"])
    base = rows[0]["t"] * rows[0]["threads"]
    print("\n### Thread scaling, one rank\n")
    print("| threads | runs | ms/step median (min-max) | GLUP/s | speedup | copy GB/s (same threads) | % of copy ceiling |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ms = lambda x: 1e3 * x / r["steps"]
        bw = stream.get(r["threads"])
        roof = f"{bw:.1f} | {100 * r['glups'] * BYTE_PER_LUP / bw:.0f}%" if bw else "n/a | n/a"
        print(f"| {r['threads']} | {r['n']} | {ms(r['t']):.1f} ({ms(r['tmin']):.1f}-{ms(r['tmax']):.1f}) | "
              f"{r['glups']:.2f} | {base / r['t'] / rows[0]['threads']:.2f} | {roof} |")


def hybrid_table(series, node_bw):
    rows = sorted((r for rows in series.values() for r in rows), key=lambda r: (r["grid"], r["ranks"]))
    print("\n### One node: MPI ranks x OpenMP threads\n")
    print("| global grid | ranks x threads | runs | ms/step median (min-max) | GLUP/s | % of copy ceiling | wait ms/step |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ms = lambda x: 1e3 * x / r["steps"]
        pct = f"{100 * r['glups'] * BYTE_PER_LUP / node_bw:.0f}%" if node_bw else "n/a"
        print(f"| {r['grid']} | {r['ranks']} x {r['threads']} | {r['n']} | {ms(r['t']):.1f} "
              f"({ms(r['tmin']):.1f}-{ms(r['tmax']):.1f}) | {r['glups']:.2f} | {pct} | {ms(r['wait']):.2f} |")


def block_table(paths):
    print("\n### y-block size (three z-planes of a block in HALO3D_BLOCK_KB), 2 ranks x 24 threads\n")
    print("| HALO3D_BLOCK_KB | rows per block (nx = 1024) | runs | ms/step median (min-max) | GLUP/s |")
    print("|---|---|---|---|---|")
    items = []
    for p in paths:
        kb = int(os.path.basename(p).split("_")[2].replace("kb.csv", ""))
        rows = [r for rows in load([p]).values() for r in rows]
        items.append((kb, rows[0]))
    for kb, r in sorted(items):
        ms = lambda x: 1e3 * x / r["steps"]
        nx = int(r["grid"].split("x")[0])
        rows_per = "whole plane" if kb == 0 else str(max(1, kb * 1024 // (3 * 8 * (nx + 2))))
        print(f"| {kb if kb else '0 (off)'} | {rows_per} | {r['n']} | {ms(r['t']):.1f} "
              f"({ms(r['tmin']):.1f}-{ms(r['tmax']):.1f}) | {r['glups']:.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strong", nargs="+", help="strong-scaling CSVs")
    ap.add_argument("--weak", nargs="+", help="weak-scaling CSVs")
    ap.add_argument("--threads", help="single-node thread-scaling CSV (one rank)")
    ap.add_argument("--stream", help="CSV threads,copy_gbs,triad_gbs from bin/bw_cpu (packed binding)")
    ap.add_argument("--roof", action="append", default=[],
                    help='"name|csv|ranks|measured GB/s|spec GB/s|FP64 GFLOP/s" (repeatable: one panel each)')
    ap.add_argument("--hybrid", help="single-node ranks x threads CSV (table only)")
    ap.add_argument("--node-bw", type=float, default=0.0, help="measured copy GB/s of one node, for --hybrid")
    ap.add_argument("--blocks", nargs="+", help="cpu_block_<KB>kb.csv files from the y-block sweep (table only)")
    ap.add_argument("--table", action="store_true", help="print markdown tables of the medians")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    strong, weak = load(a.strong), load(a.weak, per_rank=True)
    labels = sorted(set(strong) | set(weak))

    # colour and marker follow the backend (the label without the per-rank block size); the larger block in a
    # weak-scaling series is drawn dashed with hollow markers, so identity never rests on colour alone
    bases = sorted({l.split(", ")[0] + ", " + l.split(", ")[1] for l in labels})
    base_look = {b: (c, m) for b, c, m in zip(bases, COLORS, MARKERS)}
    sizes = sorted({l.rsplit(", ", 1)[1] for l in labels if l.endswith("per rank")}, key=lambda x: int(x.split("^")[0]))
    look = {}
    for l in labels:
        c, m = base_look[l.split(", ")[0] + ", " + l.split(", ")[1]]
        big = l.endswith("per rank") and len(sizes) > 1 and l.rsplit(", ", 1)[1] != sizes[0]
        look[l] = {"color": c, "marker": m, "ls": "--" if big else "-", "mfc": "white" if big else c}
    os.makedirs(a.out, exist_ok=True)
    if strong or weak:
        scaling(strong, weak, look, os.path.join(a.out, "scaling.png"))
    if strong:
        breakdown(strong, os.path.join(a.out, "breakdown_strong.png"), "Strong scaling: where each step's time goes")
    if weak:
        breakdown(weak, os.path.join(a.out, "breakdown_weak.png"), "Weak scaling: where each step's time goes")
    if a.threads:
        stream = []
        if a.stream:
            with open(a.stream, newline="") as f:
                stream = [(int(r["threads"]), float(r["copy_gbs"])) for r in csv.DictReader(f)
                          if r.get("binding", "close") == "close"]
        threads_plot(load([a.threads]), stream, os.path.join(a.out, "threads.png"))
    if a.roof:
        roofs = []
        for spec in a.roof:
            name, path, ranks, bw, sbw, fp = spec.split("|")
            roofs.append((name, path, int(ranks), float(bw), float(sbw), float(fp)))
        roofline(roofs, os.path.join(a.out, "roofline.png"))
    if a.table:
        if strong:
            table("Strong scaling", strong, "strong")
        if weak:
            table("Weak scaling", weak, "weak")
        stream = {}
        if a.stream:
            with open(a.stream, newline="") as f:
                stream = {int(r["threads"]): float(r["copy_gbs"]) for r in csv.DictReader(f)
                          if r.get("binding", "close") == "close"}
        if a.threads:
            thread_table(load([a.threads]), stream)
        if a.hybrid:
            hybrid_table(load([a.hybrid]), a.node_bw)
        if a.blocks:
            block_table(a.blocks)


if __name__ == "__main__":
    main()
