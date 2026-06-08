"""Copy + resize the selected images into the train/val/<class> folder tree.

Stores the WHOLE frame (banner included) resized so the short side is SHORT_SIDE
(downscale only), JPEG quality QUALITY. Banner cropping is done later in the
train/inference transform, not here. Resumable (skips existing destinations).

Usage:
  python copy_resize.py --test 200      # small sample, for eyeballing
  python copy_resize.py                  # full run
  python copy_resize.py --workers 24
"""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd
from PIL import Image

from label_map import OUT, IMAGE_ROOT, CLASS_ORDER

TRAIN_ROOT = r"F:\data\california-small-animals-training"
SPLIT = os.path.join(OUT, "split.parquet")
SHORT_SIDE = 512
QUALITY = 90


def process(task):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0,
                    help="process only N randomly-sampled images")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    df = pd.read_parquet(SPLIT)
    if args.test:
        df = df.sample(args.test, random_state=0)
    print(f"{len(df):,} images, {args.workers} workers -> {TRAIN_ROOT}")

    # create the 60 class dirs up front (flat layout: files go directly inside)
    for split in ("train", "val"):
        for cls in CLASS_ORDER:
            os.makedirs(os.path.join(TRAIN_ROOT, split, cls), exist_ok=True)

    tasks = [
        (os.path.join(IMAGE_ROOT, fn.replace("/", os.sep)),
         os.path.join(TRAIN_ROOT, dest.replace("/", os.sep)))
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
        log = os.path.join(OUT, "copy_resize_errors.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"wrote {len(errors)} errors -> {log}")


if __name__ == "__main__":
    main()
