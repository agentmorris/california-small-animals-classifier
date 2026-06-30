"""Spot-check: do source images exist, and what do resized outputs look like?"""
import argparse, io, os, time, random
import pandas as pd
from PIL import Image
from path_config import load_path_config

SHORT_SIDE = 512
QUALITY = 90

_ap = argparse.ArgumentParser()
_ap.add_argument("--path-config", required=True, help="JSON file of machine paths (OUT, IMAGE_ROOT)")
_cfg = load_path_config(_ap.parse_args().path_config)
OUT, IMAGE_ROOT = _cfg.OUT, _cfg.IMAGE_ROOT

df = pd.read_parquet(os.path.join(OUT, "split.parquet"))
# sample across the 3 sub-datasets and a spread of classes
df["subds"] = df.file_name.str.split("/").str[0]
print("sub-datasets:", df.subds.value_counts().to_dict())

rng = random.Random(0)
sample = df.sample(40, random_state=1)

missing = 0
out_sizes = []
t0 = time.time()
for _, r in sample.iterrows():
    src = os.path.join(IMAGE_ROOT, r.file_name.replace("/", os.sep))
    if not os.path.exists(src):
        missing += 1
        print("MISSING:", src)
        continue
    im = Image.open(src)
    w, h = im.size
    im.draft("RGB", (w, h))      # speed up JPEG decode
    im = im.convert("RGB")
    scale = SHORT_SIDE / min(w, h)
    if scale < 1:
        nw, nh = round(w * scale), round(h * scale)
        im = im.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY)
    out_sizes.append(len(buf.getvalue()))
dt = time.time() - t0

print(f"\nchecked {len(sample)} images, missing={missing}")
if out_sizes:
    import statistics as st
    kb = [s/1024 for s in out_sizes]
    print(f"resized JPEG size: min={min(kb):.0f}KB median={st.median(kb):.0f}KB "
          f"mean={st.mean(kb):.0f}KB max={max(kb):.0f}KB")
    print(f"time/image (single-thread, incl decode+resize+encode): {dt/len(sample)*1000:.0f} ms")
    total = len(df)
    print(f"\nEstimated full-set size: {st.mean(kb)*total/1024/1024:.0f} GB for {total:,} images")
    print(f"Sample resized dims e.g.: original {sample.iloc[0].width}x{sample.iloc[0].height}")
