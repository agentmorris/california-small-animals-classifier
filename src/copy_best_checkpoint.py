"""

Pick a run's best checkpoint and write a stripped, inference-ready copy to the run-folder root.

By default the best epoch is chosen by the highest ``val/acc_macro`` across the run's
``metrics*.csv`` files, and the corresponding epoch-end checkpoint
``<run_dir>/checkpoints/<run-name>-NN.ckpt`` is stripped. With ``--eval-csv`` (a CSV produced by
evaluate_all_checkpoints.py, columns checkpoint_filename, checkpoint_index, accuracy,
macro_accuracy), the best checkpoint is instead chosen from that CSV by the highest
``macro_accuracy`` (so intermediate checkpoints are eligible too), and its ``checkpoint_filename``
is taken relative to ``<run_dir>/checkpoints``.

The stripped inference checkpoint (optimizer/scheduler/callback state removed, via
strip_checkpoint.strip_one) is written to ``<run_dir>/<run-name>.best.<tag>.stripped.pt``.

Metrics-mode notes: ALL ``metrics*.csv`` files are read, not just ``metrics.csv``, because a
resumed run leaves the prior segment as a timestamped ``metrics.<ts>.csv`` (Lightning truncates
``metrics.csv`` on re-init). If an epoch appears in more than one file, the highest-``step`` row
wins, since that matches the checkpoint actually on disk; ties on score go to the later epoch.

Errors if the chosen checkpoint file is missing.

Usage:
  python copy_best_checkpoint.py <run_dir> [--eval-csv eval.csv] [--half]

"""

#%% Imports and constants

import argparse
import csv
import glob
import os

from strip_checkpoint import strip_one

METRIC = "val/acc_macro"


#%% Support functions

def best_epoch(run_dir):
    """
    Return (best_epoch, {epoch: (step, score)}, [metrics_files]) using the highest-step row
    per epoch and the max score across epochs (ties broken toward the later epoch).
    """

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


def best_from_eval_csv(csv_path):
    """
    Pick the best checkpoint from an evaluate_all_checkpoints.py CSV (columns checkpoint_filename,
    checkpoint_index, accuracy, macro_accuracy), by highest macro_accuracy, breaking ties toward
    the later checkpoint (higher checkpoint_index).

    Args:
        csv_path (str): path to the eval CSV

    Returns:
        tuple: (best_row, rows), where best_row is the winning row and rows is the full list of
        row dicts (each with a str 'checkpoint_filename', int 'checkpoint_index', and float
        'accuracy'/'macro_accuracy')
    """

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"checkpoint_filename": r["checkpoint_filename"],
                         "checkpoint_index": int(r["checkpoint_index"]),
                         "accuracy": float(r["accuracy"]),
                         "macro_accuracy": float(r["macro_accuracy"])})
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    best = max(rows, key=lambda r: (r["macro_accuracy"], r["checkpoint_index"]))
    return best, rows


#%% Command-line driver

def main():

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir",
                    help="run folder, e.g. C:\\temp\\...\\runs\\eva02-20260630-base-repro")
    ap.add_argument("--eval-csv", default=None,
                    help="an evaluate_all_checkpoints.py CSV; if given, the best checkpoint is "
                         "chosen from it (by macro_accuracy) instead of from metrics*.csv, and its "
                         "checkpoint_filename column is taken relative to <run_dir>/checkpoints")
    ap.add_argument("--half", action="store_true", help="store stripped weights as float16")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run folder not found: {run_dir}")
    run_name = os.path.basename(os.path.normpath(run_dir))
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    if args.eval_csv:
        best, rows = best_from_eval_csv(args.eval_csv)
        print(f"read {len(rows)} rows from {os.path.basename(args.eval_csv)}; "
              f"best by macro_accuracy:")
        print(f"  {best['checkpoint_filename']}  accuracy={best['accuracy']:.6f}  "
              f"macro_accuracy={best['macro_accuracy']:.6f}")
        ckpt_rel = best["checkpoint_filename"].replace("/", os.sep)
        src = os.path.join(ckpt_dir, ckpt_rel)
        stem = os.path.splitext(os.path.basename(ckpt_rel))[0]
        tag = stem[len(run_name) + 1:] if stem.startswith(run_name + "-") else stem
        score_desc = f"macro_accuracy={best['macro_accuracy']:.6f}"
    else:
        best, by_epoch, files = best_epoch(run_dir)
        nn = f"{best:02d}"
        print(f"scanned {len(files)} metrics file(s); per-epoch {METRIC}:")
        for e in sorted(by_epoch):
            print(f"  epoch {e:02d}: {by_epoch[e][1]:.6f}" + ("   <-- best" if e == best else ""))
        src = os.path.join(ckpt_dir, f"{run_name}-{nn}.ckpt")
        tag = f"epoch{nn}"
        score_desc = f"{METRIC}={by_epoch[best][1]:.6f}"

    if not os.path.isfile(src):
        raise SystemExit(f"best checkpoint missing: {src}")

    dst = os.path.join(run_dir, f"{run_name}.best.{tag}.stripped.pt")
    out, s0, s1, n, _epoch, _step = strip_one(src, {}, half=args.half, dst=dst)
    print(f"best checkpoint: {os.path.basename(src)}  ({score_desc})")
    print(f"wrote {out}")
    print(f"  {os.path.basename(src)}  {s0/1e9:.2f} GB -> {s1/1e9:.2f} GB  ({n} weight tensors)")


if __name__ == "__main__":
    main()
