"""Evaluate every checkpoint in a folder on a validation set, to trace val accuracy over training.

For each ``*.ckpt`` in the checkpoint folder this:
  1. strips it to a temporary inference checkpoint (reusing strip_checkpoint.strip_one),
  2. runs inference over the image folder (reusing run_inference.run_inference; shard across GPUs
     with --devices), writing a MegaDetector-format results file into the output folder, and
  3. scores it against the ground-truth COCO file (micro accuracy + macro / mean-per-class recall).

Outputs in the output folder:
  - ``<checkpoint-stem>.json`` per checkpoint (MegaDetector format).
  - ``accuracy_by_checkpoint.csv`` with one row per checkpoint:
      checkpoint_filename, checkpoint_index, accuracy, macro_accuracy
    ``checkpoint_index`` is -1 for last.ckpt and the 0-based training order (by global_step) for the
    rest, so sorting by it gives the chronological accuracy curve.

Resumable: a checkpoint whose results JSON already exists is re-scored without re-running inference
(delete the JSON to force a redo).

Usage:
  python evaluate_all_checkpoints.py <checkpoint-folder> <image-folder> <gt.json> <output-folder>
      [--batch-size N] [--workers N] [--precision bf16|fp32] [--classifications N] [--devices N]
"""
import argparse
import csv
import glob
import json
import os
import tempfile
from collections import defaultdict

import torch

from strip_checkpoint import strip_one
from run_inference import run_inference


def load_truth(gt_path):
    """Map val-relative file path -> true class name, from a COCO ground-truth file."""
    with open(gt_path, encoding="utf-8") as f:
        coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    file_of = {im["id"]: im["file_name"].replace("\\", "/") for im in coco["images"]}
    truth = {}
    for ann in coco["annotations"]:
        fn = file_of.get(ann["image_id"], ann["image_id"])
        truth[fn] = id_to_name[ann["category_id"]]
    return truth


def score(md_path, truth):
    """Micro and macro accuracy of an MD results file vs the truth map (compared by class name)."""
    with open(md_path, encoding="utf-8") as f:
        d = json.load(f)
    id_to_name = d["classification_categories"]
    total = defaultdict(int)
    correct = defaultdict(int)
    failures = no_truth = 0
    for im in d["images"]:
        if "failure" in im:
            failures += 1
            continue
        t = truth.get(im["file"].replace("\\", "/"))
        if t is None:
            no_truth += 1
            continue
        pred = id_to_name[im["detections"][0]["classifications"][0][0]]
        total[t] += 1
        if pred == t:
            correct[t] += 1
    n = sum(total.values())
    micro = sum(correct.values()) / n if n else 0.0
    per_class = [correct[c] / total[c] for c in total]
    macro = sum(per_class) / len(per_class) if per_class else 0.0
    return micro, macro, n, failures, no_truth


def list_checkpoints(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*.ckpt"))
                  if not f.endswith(".stripped.ckpt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_folder")
    ap.add_argument("image_folder")
    ap.add_argument("gt_file", help="ground-truth COCO json (e.g. val_cct.json)")
    ap.add_argument("output_folder")
    # passed straight through to run_inference:
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8, help="loader workers PER GPU")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--classifications", type=int, default=3,
                    help="top-N classifications per image written to the results files")
    ap.add_argument("--devices", type=int, default=1,
                    help="GPUs to shard each inference across (run_inference handles the split)")
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    truth = load_truth(args.gt_file)
    ckpts = list_checkpoints(args.checkpoint_folder)
    if not ckpts:
        raise SystemExit(f"no *.ckpt files in {args.checkpoint_folder}")
    print(f"{len(ckpts)} checkpoints | {len(truth):,} ground-truth labels | devices={args.devices}")

    cache = {}  # timm data-config cache shared across checkpoints (same model_name)
    rows = []
    for i, ckpt in enumerate(ckpts, 1):
        rel = os.path.relpath(ckpt, args.checkpoint_folder).replace("\\", "/")
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        is_last = stem.lower() == "last"
        md_out = os.path.join(args.output_folder, stem + ".json")
        print(f"\n[{i}/{len(ckpts)}] {rel}")

        # global_step (for ordering) comes from strip; if we skip inference we still need it.
        if os.path.exists(md_out):
            print(f"  results exist, re-scoring without inference: {md_out}")
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            global_step = ck.get("global_step", -1)
            del ck
        else:
            with tempfile.TemporaryDirectory(prefix="csa_eval_") as td:
                stripped = os.path.join(td, "model.stripped.ckpt")
                _dst, _s0, _s1, _n, _epoch, global_step = strip_one(ckpt, cache, dst=stripped)
                run_inference(args.image_folder, stripped, md_out,
                              batch_size=args.batch_size, workers=args.workers,
                              classifications=args.classifications,
                              precision=args.precision, devices=args.devices)

        micro, macro, n, failures, no_truth = score(md_out, truth)
        print(f"  accuracy={micro:.6f}  macro_accuracy={macro:.6f}  "
              f"(scored {n:,}, failures {failures}, no-truth {no_truth})")
        rows.append({"checkpoint_filename": rel, "is_last": is_last,
                     "global_step": global_step if global_step is not None else -1,
                     "accuracy": micro, "macro_accuracy": macro})

    # checkpoint_index: -1 for last.ckpt; others 0-based in training order (global_step, then name).
    ordered = sorted((r for r in rows if not r["is_last"]),
                     key=lambda r: (r["global_step"], r["checkpoint_filename"]))
    index_of = {r["checkpoint_filename"]: i for i, r in enumerate(ordered)}
    for r in rows:
        r["checkpoint_index"] = -1 if r["is_last"] else index_of[r["checkpoint_filename"]]

    out_csv = os.path.join(args.output_folder, "accuracy_by_checkpoint.csv")
    cols = ["checkpoint_filename", "checkpoint_index", "accuracy", "macro_accuracy"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["checkpoint_index"]):
            w.writerow(r)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
