"""Build taxonomic rollup + per-location stats for the California Small Animals dataset."""
import json
import os
import csv
from collections import Counter, defaultdict

META = r"E:\data\california-small-animals\california_small_animals_with_sequences.json"
OUT = r"C:\temp\california-small-animals-output"
os.makedirs(OUT, exist_ok=True)

def isnan(x):
    return x is None or (isinstance(x, float) and x != x) or (isinstance(x, str) and x.strip().lower() in ("", "nan"))

print("Loading...", flush=True)
with open(META, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
anns = data["annotations"]
cats = {c["id"]: c for c in data["categories"]}

# annotation counts per category
cat_counts = Counter(a["category_id"] for a in anns)

# --- Per-category table with taxonomy ---
rows = []
for cid, c in cats.items():
    rows.append({
        "id": cid,
        "name": c["name"],
        "count": cat_counts.get(cid, 0),
        "class": "" if isnan(c.get("class")) else c["class"],
        "order": "" if isnan(c.get("order")) else c["order"],
        "family": "" if isnan(c.get("family")) else c["family"],
        "genus": "" if isnan(c.get("genus")) else c["genus"],
        "species": "" if isnan(c.get("species")) else c["species"],
    })
rows.sort(key=lambda r: -r["count"])
csv_path = os.path.join(OUT, "categories.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","name","count","class","order","family","genus","species"])
    w.writeheader()
    w.writerows(rows)
print("Wrote", csv_path)

# --- Categories with NO taxonomy (special / non-taxonomic) ---
print("\n=== NON-TAXONOMIC categories (no class) ===")
for r in rows:
    if not r["class"]:
        print(f"{r['count']:>10}  {r['name']}")

# --- Rollup by class ---
def rollup(level):
    c = Counter()
    for a in anns:
        cat = cats[a["category_id"]]
        key = cat.get(level)
        if isnan(key):
            key = f"(no {level})"
        c[key] += 1
    return c

for level in ["class", "order"]:
    print(f"\n=== ROLLUP by {level} ===")
    for k, n in rollup(level).most_common():
        print(f"{n:>10}  {k}")

# --- class+order+family rollup for mammals/herps (where most action is) ---
print("\n=== ROLLUP by class/order/family ===")
cof = Counter()
for a in anns:
    cat = cats[a["category_id"]]
    cl = cat.get("class"); o = cat.get("order"); fa = cat.get("family")
    cl = "(none)" if isnan(cl) else cl
    o = "(none)" if isnan(o) else o
    fa = "(none)" if isnan(fa) else fa
    cof[(cl,o,fa)] += 1
for (cl,o,fa), n in sorted(cof.items(), key=lambda x:-x[1]):
    print(f"{n:>10}  {cl} / {o} / {fa}")

# --- Per-location stats: blanks vs animals ---
img2cat = {}
# for single-annotation images, map to the category; multi -> mark
img2anncount = Counter(a["image_id"] for a in anns)
firstcat = {}
for a in anns:
    firstcat.setdefault(a["image_id"], a["category_id"])

BLANKISH = {"blank", "misfire"}
loc_total = Counter()
loc_blank = Counter()
for im in images:
    loc = im["location"]
    loc_total[loc] += 1
    cid = firstcat.get(im["id"])
    nm = cats[cid]["name"] if cid is not None else None
    if nm in BLANKISH:
        loc_blank[loc] += 1

print(f"\n=== PER-LOCATION (701 locations) ===")
totals = sorted(loc_total.values())
print(f"images/location: min={totals[0]} median={totals[len(totals)//2]} max={totals[-1]}")
print(f"total blank+misfire images: {sum(loc_blank.values())}")
# locations with very few images
few = sum(1 for v in loc_total.values() if v < 200)
print(f"locations with <200 images: {few}")

# save per-location counts
with open(os.path.join(OUT, "per_location.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["location","total","blank_misfire"])
    for loc in sorted(loc_total):
        w.writerow([loc, loc_total[loc], loc_blank[loc]])
print("Wrote per_location.csv")

# --- Image size distribution ---
sizes = Counter((im["width"], im["height"]) for im in images)
print("\n=== IMAGE SIZES (top 10) ===")
for (w,h), n in sizes.most_common(10):
    print(f"{n:>10}  {w}x{h}")

# --- Multi-annotation breakdown: what combos ---
multi_imgs = [iid for iid, n in img2anncount.items() if n > 1]
combo = Counter()
img2cats = defaultdict(list)
for a in anns:
    img2cats[a["image_id"]].append(cats[a["category_id"]]["name"])
for iid in multi_imgs:
    combo[tuple(sorted(set(img2cats[iid])))] += 1
print(f"\n=== MULTI-ANNOTATION combos (top 15 of {len(multi_imgs)}) ===")
for c, n in combo.most_common(15):
    print(f"{n:>6}  {c}")

print("\nDONE")
