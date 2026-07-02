"""
Copy + resize the selected images into the train/val/<class> folder tree.

Stores the WHOLE frame (banner included) resized so the short side is SHORT_SIDE
(downscale only), JPEG quality QUALITY. Banner cropping is done later in the
train/inference transform, not here. Resumable (skips existing destinations).

Usage:
  python copy_resize.py --test 200      # small sample, for eyeballing
  python copy_resize.py                  # full run
  python copy_resize.py --workers 24
"""

#%% Imports and constants

import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd
from PIL import Image

from label_map import CLASS_ORDER
from path_config import load_path_config, load_excluded_guids

SHORT_SIDE = 512
QUALITY = 90


#%% Support functions

def process(task):
    """
    Resize one image
    """

    src, dst = task
    if os.path.exists(dst):
        return ("skip", dst)
    try:
        im = Image.open(src)
        w, h = im.size
        short = min(w, h)
        if short > SHORT_SIDE:
            scale = SHORT_SIDE / short
            tw, th = round(w * scale), round(h * scale)
            im.draft("RGB", (tw, th))               # fast partial JPEG decode
            im = im.convert("RGB").resize((tw, th), Image.LANCZOS)
        else:
            im = im.convert("RGB")
        im.save(dst, "JPEG", quality=QUALITY)
        return ("ok", dst)
    except Exception as e:
        return ("err", f"{src}\t{type(e).__name__}: {e}")


#%% Command-line driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-config", required=True,
                    help="JSON file of machine paths (OUTPUT_ROOT, IMAGE_ROOT, TRAIN_ROOT, "
                         "EXCLUDE_FILES)")
    ap.add_argument("--test", type=int, default=0,
                    help="process only N randomly-sampled images")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--exclude", action="append", default=None,
                    help="manual-review JSON(s); images marked 'incorrect' are dropped "
                         "(default: the path-config's EXCLUDE_FILES)")
    ap.add_argument("--no-exclude", action="store_true",
                    help="ignore manual-review exclusions (copy every image in the split)")
    args = ap.parse_args()

    cfg = load_path_config(args.path_config)
    image_root, train_root = cfg.IMAGE_ROOT, cfg.TRAIN_ROOT
    df = pd.read_parquet(os.path.join(cfg.OUTPUT_ROOT, "split.parquet"))

    exclude_paths = [] if args.no_exclude else (args.exclude or cfg.EXCLUDE_FILES)
    if exclude_paths:
        excluded = load_excluded_guids(exclude_paths)
        before = len(df)
        df = df[~df.image_id.isin(excluded)]
        print(f"manual-review: dropped {before - len(df):,} images ({len(excluded):,} excluded guids)")

    if args.test:
        df = df.sample(args.test, random_state=0)
    print(f"{len(df):,} images, {args.workers} workers -> {train_root}")

    # create the class dirs up front (flat layout: files go directly inside)
    for split in ("train", "val"):
        for cls in CLASS_ORDER:
            os.makedirs(os.path.join(train_root, split, cls), exist_ok=True)

    tasks = [
        (os.path.join(image_root, fn.replace("/", os.sep)),
         os.path.join(train_root, dest.replace("/", os.sep)))
        for fn, dest in zip(df.file_name, df.dest_rel)
    ]

    counts = {"ok": 0, "skip": 0, "err": 0}
    errors = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(process, tasks, chunksize=64), 1):
            counts[status] += 1
            if status == "err":
                errors.append(info)
            if i % 20000 == 0:
                rate = i / (time.time() - t0)
                eta = (len(tasks) - i) / rate / 60
                print(f"  {i:,}/{len(tasks):,}  {rate:.0f} img/s  ETA {eta:.0f} min  "
                      f"ok={counts['ok']:,} skip={counts['skip']:,} err={counts['err']:,}",
                      flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min: ok={counts['ok']:,} skip={counts['skip']:,} "
          f"err={counts['err']:,}  ({len(tasks)/dt:.0f} img/s)")
    if errors:
        log = os.path.join(cfg.OUTPUT_ROOT, "copy_resize_errors.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"wrote {len(errors)} errors -> {log}")

# ...def main(...)

if __name__ == "__main__":
    main()
