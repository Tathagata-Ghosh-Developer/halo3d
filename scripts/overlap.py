#!/usr/bin/env python3
"""Per-step overlap numbers from Nsight Systems GPU traces (nsys stats --report cuda_gpu_trace --format csv).

  python3 scripts/overlap.py --trace "before fix=results/nsys/v1_rank0_cuda_gpu_trace.csv" \
                             --trace "after fix=results/nsys/v2_rank0_cuda_gpu_trace.csv" --out results

Every step ends with the shell kernels, so the trace is split into steps after each run of shell kernels.
Per step (median over steps): the interior-kernel time; the step period (start of this step to start of the
next, idle GPU time included); when the device-to-host staging of the faces finished and when the received
faces came back (host-to-device copy start, i.e. MPI done), both relative to the interior-kernel start; and
how long the last shell kernel ended after the interior kernel. Writes timeline.png (one step per trace).
"""
import argparse
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"interior": "#2a78d6", "pack/unpack": "#eb6834", "shell": "#1baf7a", "D2H": "#eda100", "H2D": "#4a3aa7"}


def kind(name):
    n = name.lower()
    if "interior_kernel" in n:
        return "interior"
    if "face_kernel" in n:
        return "pack/unpack"
    if "shell_kernel" in n:
        return "shell"
    if "memcpy" in n and ("dtoh" in n or "device-to-host" in n):
        return "D2H"
    if "memcpy" in n and ("htod" in n or "host-to-device" in n):
        return "H2D"
    return None


def steps(path):
    ev = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = kind(r["Name"])
            if k:
                b = float(r["Start (ns)"]) * 1e-6
                ev.append((b, b + float(r["Duration (ns)"]) * 1e-6, k))
    ev.sort()
    out, cur = [], []
    for e in ev:
        if cur and cur[-1][2] == "shell" and e[2] != "shell" and any(x[2] == "interior" for x in cur):
            out.append(cur)
            cur = []
        cur.append(e)
    return out   # the last group also holds the final copy-out and is dropped by the caller


def metrics(step, next_step):
    i0, i1 = next((e[0], e[1]) for e in step if e[2] == "interior")
    t0 = min(e[0] for e in step)
    d2h = [e[1] for e in step if e[2] == "D2H"]
    h2d = [e[0] for e in step if e[2] == "H2D"]
    shell = [e[1] for e in step if e[2] == "shell"]
    return {"interior": i1 - i0, "period": min(e[0] for e in next_step) - t0,
            "d2h_done": max(d2h) - i0, "h2d_start": min(h2d) - i0, "shell_end": max(shell) - i1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", action="append", required=True, help='"label=path to cuda_gpu_trace csv"')
    ap.add_argument("--skip", type=int, default=2, help="warm-up steps to ignore")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs = [(t.split("=", 1)[0], steps(t.split("=", 1)[1])) for t in a.trace]
    print("| trace | steps | interior kernel | step period | faces staged to host (D2H done) | "
          "received faces back (H2D starts) | last shell kernel ends after interior |")
    print("|---|---|---|---|---|---|---|")
    for label, ss in runs:
        m = [metrics(s, n) for s, n in zip(ss[a.skip:-1], ss[a.skip + 1:])]
        med = {k: st.median(x[k] for x in m) for k in m[0]}
        print(f"| {label} | {len(m)} | {med['interior']:.2f} ms | {med['period']:.2f} ms | "
              f"{med['d2h_done']:.2f} ms | {med['h2d_start']:.2f} ms | {med['shell_end']:.2f} ms |")
    print("(times in the last three columns are measured from the start of the interior kernel, "
          "except the last, which is measured from its end)")
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 1.9 * len(runs) + 0.6), squeeze=False)
    for ax, (label, ss) in zip(axes[:, 0], runs):
        s = ss[min(a.skip + 3, len(ss) - 1)]
        t0 = min(e[0] for e in s)
        lanes = {"interior": 1, "pack/unpack": 0, "shell": 0, "D2H": 0, "H2D": 0}
        seen = set()
        for b, e, k in s:
            ax.broken_barh([(b - t0, max(e - b, 0.004))], (lanes[k] - 0.35, 0.7), color=COLORS[k],
                           label=None if k in seen else k)
            seen.add(k)
        ax.set_yticks([0, 1], ["halo stream\n(high priority after fix)", "interior stream"])
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(True, axis="x", lw=0.5, alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper right", bbox_to_anchor=(1.0, 1.32))
    axes[-1, 0].set_xlabel("ms since the step's first GPU operation (rank 0 of 4 GPUs on 2 nodes, 1024^3)")
    xmax = max(ax.get_xlim()[1] for ax in axes[:, 0])
    for ax in axes[:, 0]:
        ax.set_xlim(0, xmax)
    fig.tight_layout()
    path = os.path.join(a.out, "timeline.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
