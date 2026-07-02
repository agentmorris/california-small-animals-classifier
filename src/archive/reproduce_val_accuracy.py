"""

Recompute val accuracy from a MegaDetector-format predictions file and check it
against the numbers logged during training (metrics.csv).

True labels come from the val folder structure: inference is run on the val/ tree,
so each `file` is "<class>/<image>.jpg" and the first path component is the truth.
Computes micro accuracy (overall) and macro accuracy (mean per-class recall), to
match torchmetrics MulticlassAccuracy(average="micro"/"macro") used in training.

Usage:
  python reproduce_val_accuracy.py <predictions.json> [--metrics-csv metrics.csv] [--epoch N]

"""

#%% Imports and constants

import argparse
import json
from collections import defaultdict

import pandas as pd


#%% Command-line driver
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="MegaDetector-format predictions JSON (run on val/)")
    ap.add_argument("--metrics-csv", default=None, help="training metrics.csv to compare against")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch row to compare (default: best val/acc_macro)")
    args = ap.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        d = json.load(f)
    id2name = d["classification_categories"]
    valid = set(id2name.values())

    total = defaultdict(int)
    correct = defaultdict(int)
    failures = 0
    bad_truth = 0
    for im in d["images"]:
        if "failure" in im:
            failures += 1
            continue
        true = im["file"].split("/")[0]
        if true not in valid:
            bad_truth += 1
            continue
        pred = id2name[im["detections"][0]["classifications"][0][0]]
        total[true] += 1
        if pred == true:
            correct[true] += 1

    n = sum(total.values())
    n_correct = sum(correct.values())
    micro = n_correct / n if n else 0.0
    per_class = {c: correct[c] / total[c] for c in total}
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0

    print(f"images scored: {n:,}  (failures: {failures}, unrecognized-truth: {bad_truth})")
    print(f"\n=== per-class accuracy (recall) ===")
    for c in sorted(per_class, key=per_class.get):
        print(f"  {c:<26} {per_class[c]*100:5.1f}%   ({correct[c]:,}/{total[c]:,})")
    print(f"\nMICRO (overall) accuracy: {micro:.6f}")
    print(f"MACRO (mean per-class)  : {macro:.6f}")

    if args.metrics_csv:
        m = pd.read_csv(args.metrics_csv)
        v = m.dropna(subset=["val/acc_macro"])[["epoch", "val/acc_micro", "val/acc_macro"]]
        if args.epoch is not None:
            row = v[v.epoch == args.epoch].iloc[0]
        else:
            row = v.loc[v["val/acc_macro"].idxmax()]
        print(f"\n=== metrics.csv (epoch {int(row.epoch)}) ===")
        print(f"  val/acc_micro: {row['val/acc_micro']:.6f}   "
              f"(inference {micro:.6f}, diff {micro-row['val/acc_micro']:+.6f})")
        print(f"  val/acc_macro: {row['val/acc_macro']:.6f}   "
              f"(inference {macro:.6f}, diff {macro-row['val/acc_macro']:+.6f})")


if __name__ == "__main__":
    main()
