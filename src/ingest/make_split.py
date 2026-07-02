"""

Class-aware 85/15 train/val split BY CAMERA, plus blank downsampling.

- Every camera goes entirely to train or val (no camera in both).
- Greedy, rarest-class-first assignment so each class lands ~15% in val and is
  present on both sides.
- Blank images are downsampled to 1 frame per sequence, then capped at 300 per
  camera (applied per-camera, so it carries through the split).

Outputs:
  split.parquet          per kept image: + 'split' (train/val) + 'dest_rel' path
  camera_split.csv       camera -> split assignment (locked / documented)

"""

#%% Imports and constants

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pulp

from label_map import CLASS_ORDER
from path_config import load_path_config

VAL_FRAC = 0.15
BLANK_FRAMES_PER_SEQ = 1
BLANK_CAP_PER_CAM = 300
SEED = 20260608


#%% Split assignment functions

def assign_split(df):
    """
    ILP camera->{train,val}: minimize per-class RELATIVE deviation from VAL_FRAC.

    Each camera is one binary var x_cam (1 = val). For each class we penalize
    |val_images_c - VAL_FRAC * total_c| weighted by 1/total_c, so a 1k-image class
    and a 250k-image class are both pushed toward 15% in relative terms. Rare
    classes can't hit exactly 15% (lumpy), but this gets as close as possible.
    """

    cc = (df.groupby(["location", "target_class"], observed=True)
            .size().unstack(fill_value=0))
    cameras = list(cc.index)
    classes = [c for c in CLASS_ORDER if c in cc.columns]
    class_total = {c: int(cc[c].sum()) for c in classes}

    prob = pulp.LpProblem("camera_split", pulp.LpMinimize)
    x = {cam: pulp.LpVariable(f"x_{i}", cat="Binary") for i, cam in enumerate(cameras)}

    obj = []
    for c in classes:
        target = VAL_FRAC * class_total[c]
        val_imgs = pulp.lpSum(int(cc.at[cam, c]) * x[cam]
                              for cam in cameras if cc.at[cam, c] > 0)
        dev = pulp.LpVariable(f"dev_{c}", lowBound=0)
        prob += dev >= val_imgs - target
        prob += dev >= target - val_imgs
        obj.append(dev / class_total[c])           # relative deviation
    prob += pulp.lpSum(obj)

    # keep overall val fraction in a sane band (belt-and-suspenders)
    total_imgs = sum(class_total.values())
    all_val = pulp.lpSum(int(cc.loc[cam].sum()) * x[cam] for cam in cameras)
    prob += all_val >= 0.13 * total_imgs
    prob += all_val <= 0.17 * total_imgs

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
    status = pulp.LpStatus[prob.status]
    print(f"ILP status: {status}")
    split = {cam: ("val" if x[cam].value() and x[cam].value() > 0.5 else "train")
             for cam in cameras}

    return split

# def assign_split(...)

def downsample_blanks(df):
    """
    Keep BLANK_FRAMES_PER_SEQ earliest frame(s) per blank sequence, then cap
    BLANK_CAP_PER_CAM per camera. Returns index of blank rows to KEEP.
    """

    blank = df[df.target_class == "blank"]
    # earliest frame(s) per sequence
    b = blank.sort_values(["seq_id", "frame_num", "image_id"])
    per_seq = b.groupby("seq_id", observed=True).head(BLANK_FRAMES_PER_SEQ)
    # cap per camera (deterministic order by seq_id)
    per_seq = per_seq.sort_values(["location", "seq_id"])
    capped = per_seq.groupby("location", observed=True).head(BLANK_CAP_PER_CAM)
    return set(capped.index)


#%% Command-line driver

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--path-config", required=True, help="JSON file of machine paths (OUTPUT_ROOT)")
    args = ap.parse_args()
    OUT = load_path_config(args.path_config).OUTPUT_ROOT

    df = pd.read_parquet(os.path.join(OUT, "manifest.parquet"))
    df["target_class"] = df["target_class"].astype(str)
    print(f"manifest: {len(df):,} images, {df.location.nunique()} cameras")

    split_map = assign_split(df)
    df["split"] = df.location.map(split_map)

    keep_blank = downsample_blanks(df)
    is_blank = df.target_class == "blank"
    keep_mask = (~is_blank) | df.index.isin(keep_blank)
    kept = df[keep_mask].copy()

    # destination relative path: <split>/<class>/<location>__<image_id>.jpg
    # (flatten but keep camera in the filename for debugging / relative-path preference)
    kept["dest_rel"] = (kept["split"] + "/" + kept["target_class"] + "/"
                        + kept["location"].astype(str) + "__"
                        + kept["image_id"].astype(str) + ".jpg")

    kept.to_parquet(os.path.join(OUT, "split.parquet"), index=False)

    cam_df = pd.DataFrame(
        sorted(split_map.items()), columns=["location", "split"])
    cam_df.to_parquet(os.path.join(OUT, "camera_split.parquet"), index=False)
    cam_df.to_csv(os.path.join(OUT, "camera_split.csv"), index=False)

    # ---- report ----
    n_val_cam = sum(1 for v in split_map.values() if v == "val")
    print(f"cameras: train={len(split_map)-n_val_cam}  val={n_val_cam}")
    print(f"kept images: {len(kept):,} (blank downsampled "
          f"{is_blank.sum():,} -> {len(keep_blank):,})")

    tab = (kept.groupby(["target_class", "split"], observed=True).size()
           .unstack(fill_value=0).reindex(CLASS_ORDER))
    tab["total"] = tab.sum(axis=1)
    tab["val_%"] = (100 * tab["val"] / tab["total"]).round(1)
    print("\n=== per-class train/val ===")
    print(tab.to_string())
    print(f"\nTOTAL kept: {len(kept):,}   "
          f"overall val fraction: {100*(kept.split=='val').mean():.1f}%")


if __name__ == "__main__":
    main()
