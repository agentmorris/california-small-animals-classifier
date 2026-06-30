"""Emit a self-contained provenance record of the train/val split and the
source-category -> training-class assignments.

Writes training_info.<YYYYMMDD>.json into the repo. Reproducible from the fixed
metadata + the split artifacts in the output folder.
"""
import argparse
import json
import os
from collections import defaultdict, OrderedDict

import pandas as pd

from label_map import CLASS_ORDER, target_class
from path_config import load_path_config
from make_split import VAL_FRAC, BLANK_FRAMES_PER_SEQ, BLANK_CAP_PER_CAM, SEED

DATE = "20260608"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ -> repo root
OUT_JSON = os.path.join(REPO_ROOT, f"training_info.{DATE}.json")

EXCLUDE_REASONS = {
    "unknown": "not a coherent visual class",
    "animal": "too generic (any animal)",
    "no cv result": "not a real label",
    "lizards and snakes": "ambiguous between lizard and snake",
    "reptile": "too generic / ambiguous lizard-vs-snake",
    "western pond turtle": "turtle (testudines); too few, not a target class",
    "pond slider": "turtle (testudines); too few, not a target class",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-config", required=True, help="JSON file of machine paths (META, OUT)")
    args = ap.parse_args()
    cfg = load_path_config(args.path_config)
    META, OUT = cfg.META, cfg.OUT

    with open(META, encoding="utf-8") as f:
        data = json.load(f)
    cats = data["categories"]

    # annotation count per source category
    acount = defaultdict(int)
    for a in data["annotations"]:
        acount[a["category_id"]] += 1

    # source -> target
    src_rows = []
    for c in cats:
        src_rows.append((c["name"], acount.get(c["id"], 0), target_class(c)))

    # flat map + grouped
    source_to_class = OrderedDict(
        (name, tgt) for name, _, tgt in sorted(src_rows, key=lambda r: -r[1]))
    grouped = defaultdict(list)
    for name, n, tgt in src_rows:
        grouped[tgt].append({"source_category": name, "annotations": n})
    for tgt in grouped:
        grouped[tgt].sort(key=lambda d: -d["annotations"])

    category_assignments = OrderedDict()
    for tgt in CLASS_ORDER + ["EXCLUDE"]:
        srcs = grouped.get(tgt, [])
        entry = {
            "n_source_categories": len(srcs),
            "total_annotations": sum(s["annotations"] for s in srcs),
            "sources": srcs,
        }
        if tgt == "EXCLUDE":
            for s in entry["sources"]:
                s["reason"] = EXCLUDE_REASONS.get(s["source_category"], "excluded")
        category_assignments[tgt] = entry

    # camera split
    cam = pd.read_parquet(os.path.join(OUT, "camera_split.parquet"))
    cam = cam.sort_values("location")
    camera_split = OrderedDict(zip(cam.location.astype(str), cam.split.astype(str)))
    n_train_cam = int((cam.split == "train").sum())
    n_val_cam = int((cam.split == "val").sum())

    # per-class kept-image counts by split (actual training pool, post blank downsample)
    sp = pd.read_parquet(os.path.join(OUT, "split.parquet"),
                         columns=["target_class", "split"])
    sp["target_class"] = sp["target_class"].astype(str)
    tab = sp.groupby(["target_class", "split"]).size().unstack(fill_value=0)
    per_class = OrderedDict()
    for c in CLASS_ORDER:
        tr = int(tab.loc[c, "train"]) if c in tab.index and "train" in tab else 0
        va = int(tab.loc[c, "val"]) if c in tab.index and "val" in tab else 0
        per_class[c] = {"train": tr, "val": va, "total": tr + va,
                        "val_pct": round(100 * va / (tr + va), 1) if tr + va else 0}

    info = OrderedDict([
        ("created", "2026-06-08"),
        ("dataset", "california-small-animals"),
        ("source_metadata", os.path.basename(META)),
        ("description",
         "First-pass classifier: flat 29-class label set (27 animal + blank + "
         "setup_pickup); split by camera (location==folder, 1:1). This file records "
         "the train/val camera assignment and how the 256 source categories were "
         "mapped/merged/excluded."),
        ("parameters", OrderedDict([
            ("split_method", "class-aware ILP by camera, minimize per-class "
                             "relative deviation from val_fraction"),
            ("val_fraction", VAL_FRAC),
            ("split_seed", SEED),
            ("blank_downsampling", {"frames_per_sequence": BLANK_FRAMES_PER_SEQ,
                                    "per_camera_cap": BLANK_CAP_PER_CAM}),
            ("multi_annotation_images", "dropped (9,372)"),
            ("image_resize", "whole frame, short side ~512px, JPEG q90"),
        ])),
        ("n_training_classes", len(CLASS_ORDER)),
        ("training_classes",
         [{"index": i, "name": c} for i, c in enumerate(CLASS_ORDER)]),
        ("per_class_image_counts", per_class),
        ("category_assignments", category_assignments),
        ("source_category_to_training_class", source_to_class),
        ("camera_split_summary", {"n_cameras": len(camera_split),
                                  "n_train_cameras": n_train_cam,
                                  "n_val_cameras": n_val_cam}),
        ("camera_split", camera_split),
    ])

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")
    print(f"  {len(CLASS_ORDER)} classes, {len(source_to_class)} source categories, "
          f"{category_assignments['EXCLUDE']['n_source_categories']} excluded")
    print(f"  cameras: {n_train_cam} train / {n_val_cam} val")


if __name__ == "__main__":
    main()
