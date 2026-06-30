"""Build the per-image manifest (single source of truth for split + copy) and
report blank/sequence/per-camera statistics.

Keeps only single-annotation images whose category maps to a real class
(EXCLUDE and multi-annotation images are dropped). Writes manifest.parquet.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import pandas as pd

from label_map import CLASS_ORDER, target_class
from path_config import load_path_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-config", required=True,
                    help="JSON file of machine paths (METADATA_FILE, OUTPUT_ROOT)")
    args = ap.parse_args()
    cfg = load_path_config(args.path_config)
    META, OUT = cfg.METADATA_FILE, cfg.OUTPUT_ROOT
    MANIFEST = os.path.join(OUT, "manifest.parquet")

    print("Loading fixed JSON...", flush=True)
    with open(META, encoding="utf-8") as f:
        data = json.load(f)

    cats = {c["id"]: c for c in data["categories"]}
    src2tgt = {cid: target_class(c) for cid, c in cats.items()}

    # count annotations per image; remember the (single) annotation's category
    anncount = Counter()
    firstcat = {}
    for a in data["annotations"]:
        iid = a["image_id"]
        anncount[iid] += 1
        if iid not in firstcat:
            firstcat[iid] = a["category_id"]

    rows = []
    dropped_multi = 0
    dropped_excl = 0
    for im in data["images"]:
        iid = im["id"]
        if anncount[iid] != 1:
            dropped_multi += 1
            continue
        tgt = src2tgt[firstcat[iid]]
        if tgt == "EXCLUDE":
            dropped_excl += 1
            continue
        fn = im["file_name"].replace("\\", "/")
        rows.append((
            iid, fn, im["location"], im.get("seq_id", ""),
            im.get("frame_num", -1), im["width"], im["height"], tgt,
        ))

    df = pd.DataFrame(rows, columns=[
        "image_id", "file_name", "location", "seq_id",
        "frame_num", "width", "height", "target_class",
    ])
    # categorical encoding keeps the parquet small
    df["target_class"] = pd.Categorical(df["target_class"], categories=CLASS_ORDER)
    df["location"] = df["location"].astype("category")
    df.to_parquet(MANIFEST, index=False)
    print(f"\nWrote {MANIFEST}  ({len(df):,} rows)")
    print(f"dropped multi-annotation: {dropped_multi:,}   dropped EXCLUDE: {dropped_excl:,}")

    # ---------- per-class ----------
    print("\n=== images per class ===")
    for c in CLASS_ORDER:
        print(f"{c:<26}{(df.target_class == c).sum():>12,}")

    # ---------- blank stats ----------
    blank = df[df.target_class == "blank"]
    print("\n=== BLANK downsampling inputs ===")
    print(f"blank images:            {len(blank):,}")
    print(f"distinct blank seq_id:   {blank.seq_id.nunique():,}")
    print(f"cameras with any blank:  {blank.location.nunique():,}")
    bl_per_cam = blank.groupby("location", observed=True).size().sort_values()
    print(f"blank images/camera:     min={bl_per_cam.min()} "
          f"median={int(bl_per_cam.median())} max={bl_per_cam.max()}")
    seq_per_cam = blank.groupby("location", observed=True).seq_id.nunique().sort_values()
    print(f"blank sequences/camera:  min={seq_per_cam.min()} "
          f"median={int(seq_per_cam.median())} max={seq_per_cam.max()}")
    fps = blank.groupby("seq_id", observed=True).size()
    print(f"frames per blank seq:    min={fps.min()} median={int(fps.median())} "
          f"mean={fps.mean():.1f} max={fps.max()}")
    # how many blanks if we keep 1 per sequence, then cap per camera at various N
    one_per_seq = blank.drop_duplicates(["location", "seq_id"])
    print(f"\nblanks if 1 frame/sequence:        {len(one_per_seq):,}")
    for cap in (100, 200, 300, 500):
        capped = one_per_seq.groupby("location", observed=True).head(cap)
        print(f"  + cap {cap:>4}/camera -> {len(capped):>9,} blanks "
              f"({capped.location.nunique()} cameras)")

    # ---------- per-camera class coverage (for the split) ----------
    nonblank = df[df.target_class != "blank"]
    # classes x #cameras they appear on
    cls_cams = nonblank.groupby("target_class", observed=True).location.nunique()
    print("\n=== #cameras each non-blank class appears on (split feasibility) ===")
    for c in CLASS_ORDER:
        if c == "blank":
            continue
        n = int(cls_cams.get(c, 0))
        flag = "  <-- single-camera!" if n == 1 else ("  <-- few cameras" if n <= 3 else "")
        print(f"{c:<26}{n:>5} cameras{flag}")

    # save per (location, class) counts for the splitter
    piv = (df.groupby(["location", "target_class"], observed=True)
             .size().rename("n").reset_index())
    piv.to_parquet(os.path.join(OUT, "camera_class_counts.parquet"), index=False)
    print("\nWrote camera_class_counts.parquet")


if __name__ == "__main__":
    main()
