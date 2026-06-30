"""Pick a run's best checkpoint and write a stripped, inference-ready copy to the run-folder root.

Finds the epoch with the highest ``val/acc_macro`` across the run's metrics, then strips the
optimizer/scheduler/callback state from that epoch's checkpoint (via strip_checkpoint.strip_one)
and writes it to ``runs/<run-name>/<run-name>.best.epochNN.stripped.pt``.

Metrics are read from ALL ``metrics*.csv`` files in the run folder, not just ``metrics.csv``: a
resumed run leaves the prior segment as a timestamped ``metrics.<ts>.csv`` (Lightning truncates
``metrics.csv`` on re-init), so the per-epoch history is spread across multiple files. If an epoch
appears in more than one file (an abandoned attempt plus its post-resume redo), the row with the
highest ``step`` wins, since that matches the checkpoint actually on disk.

Errors if the chosen epoch's checkpoint file is missing.

Usage:
  python copy_best_checkpoint.py <run-name> [--half]
"""
import argparse
import csv
import glob
import os

from path_config import load_path_config
from strip_checkpoint import strip_one

METRIC = "val/acc_macro"


def best_epoch(run_dir):
    """Return (best_epoch, {epoch: (step, score)}, [metrics_files]) using the highest-step row
    per epoch and the max score across epochs (ties broken toward the later epoch)."""
    files = sorted(glob.glob(os.path.join(run_dir, "metrics*.csv")))
    if not files:
        raise SystemExit(f"no metrics*.csv files in {run_dir}")

    by_epoch = {}  # epoch -> (step, score), keeping the highest-step (most recent) row
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                v = r.get(METRIC)
                if not v:
                    continue
                ep, step, score = int(r["epoch"]), int(r["step"]), float(v)
                if ep not in by_epoch or step > by_epoch[ep][0]:
                    by_epoch[ep] = (step, score)
    if not by_epoch:
        raise SystemExit(f"no '{METRIC}' values found in: {', '.join(os.path.basename(f) for f in files)}")

    best = max(by_epoch, key=lambda e: (by_epoch[e][1], e))
    return best, by_epoch, files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name")
    ap.add_argument("--path-config",
                    help="JSON file of machine paths; the run folder is <OUT>/runs/<run-name>. "
                         "Omit only if --runs-dir is given.")
    ap.add_argument("--runs-dir", default=None,
                    help="override: directory holding run folders (default: <OUT>/runs)")
    ap.add_argument("--half", action="store_true", help="store stripped weights as float16")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if runs_dir is None:
        if not args.path_config:
            raise SystemExit("provide --path-config (for <OUT>/runs) or --runs-dir")
        runs_dir = os.path.join(load_path_config(args.path_config).OUT, "runs")

    run_dir = os.path.join(runs_dir, args.run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run folder not found: {run_dir}")

    best, by_epoch, files = best_epoch(run_dir)
    nn = f"{best:02d}"
    print(f"scanned {len(files)} metrics file(s); per-epoch {METRIC}:")
    for e in sorted(by_epoch):
        print(f"  epoch {e:02d}: {by_epoch[e][1]:.6f}" + ("   <-- best" if e == best else ""))

    src = os.path.join(run_dir, "checkpoints", f"{args.run_name}-{nn}.ckpt")
    if not os.path.isfile(src):
        raise SystemExit(f"best checkpoint missing: {src}")

    dst = os.path.join(run_dir, f"{args.run_name}.best.epoch{nn}.stripped.pt")
    out, s0, s1, n, _epoch, _step = strip_one(src, {}, half=args.half, dst=dst)
    print(f"best epoch {nn}  ({METRIC}={by_epoch[best][1]:.6f})")
    print(f"wrote {out}")
    print(f"  {os.path.basename(src)}  {s0/1e9:.2f} GB -> {s1/1e9:.2f} GB  ({n} weight tensors)")


if __name__ == "__main__":
    main()
