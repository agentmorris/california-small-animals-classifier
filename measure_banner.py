"""Measure the Reconyx info-banner band heights (top & bottom) across a sample."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from label_map import OUT, IMAGE_ROOT

df = pd.read_parquet(os.path.join(OUT, "split.parquet"))
df["subds"] = df.file_name.str.split("/").str[0]

def band_heights(arr):
    """Return (top_px, bottom_px) of the dark banner bands.
    Banner rows are mostly near-black with sparse bright text; scene rows have
    lots of mid-gray. Detect contiguous 'banner-like' rows from each edge."""
    h, w = arr.shape
    blackfrac = (arr < 40).mean(axis=1)      # fraction near-black per row
    brightfrac = (arr > 200).mean(axis=1)    # fraction near-white (text) per row
    # a banner row: very dark background, maybe a little bright text
    is_banner = (blackfrac > 0.55) & (brightfrac < 0.25)
    top = 0
    for i in range(min(h // 5, h)):
        if is_banner[i]:
            top = i + 1
        else:
            if i - top > 3:   # allow a couple non-banner rows inside text gaps
                break
    bot = 0
    for j in range(min(h // 5, h)):
        if is_banner[h - 1 - j]:
            bot = j + 1
        else:
            if j - bot > 3:
                break
    return top, bot

rows = []
for subds, g in df.groupby("subds"):
    samp = g.sample(min(15, len(g)), random_state=2)
    for _, r in samp.iterrows():
        p = os.path.join(IMAGE_ROOT, r.file_name.replace("/", os.sep))
        try:
            im = Image.open(p).convert("L")
        except Exception as e:
            continue
        arr = np.asarray(im)
        t, b = band_heights(arr)
        rows.append((subds, arr.shape[1], arr.shape[0], t, b,
                     round(100*t/arr.shape[0], 1), round(100*b/arr.shape[0], 1)))

res = pd.DataFrame(rows, columns=["subds", "W", "H", "top_px", "bot_px", "top_%", "bot_%"])
print(res.groupby(["subds", "W", "H"]).agg(
    n=("top_px", "size"),
    top_px_med=("top_px", "median"), top_px_max=("top_px", "max"),
    bot_px_med=("bot_px", "median"), bot_px_max=("bot_px", "max"),
    top_pct_med=("top_%", "median"), bot_pct_med=("bot_%", "median"),
).to_string())
print("\noverall top_% median/max:", res["top_%"].median(), res["top_%"].max())
print("overall bot_% median/max:", res["bot_%"].median(), res["bot_%"].max())
