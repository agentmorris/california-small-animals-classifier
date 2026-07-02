"""

Analyze the California Small Animals COCO Camera Traps metadata.

"""

#%% Headers and constants

import json
import os
from collections import Counter, defaultdict

metadata_file = r"E:\data\california-small-animals\california_small_animals_with_sequences.json"


#%% Analyze metadata

print("Loading JSON (1.3GB, give it a minute)...", flush=True)
with open(metadata_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
for k in data:
    if isinstance(data[k], list):
        print(f"  {k}: list of {len(data[k])}")
    else:
        print(f"  {k}: {type(data[k]).__name__}")

images = data["images"]
anns = data["annotations"]
cats = data["categories"]

print("\n=== SAMPLE IMAGE ===")
print(json.dumps(images[0], indent=2))
print("\n=== SAMPLE ANNOTATION ===")
print(json.dumps(anns[0], indent=2))
print("\n=== SAMPLE CATEGORIES (first 5) ===")
for c in cats[:5]:
    print(json.dumps(c, indent=2))

print(f"\n#images={len(images)}  #annotations={len(anns)}  #categories={len(cats)}")

# Category id -> name
catid2name = {c["id"]: c["name"] for c in cats}

# Annotations per image
img2anns = defaultdict(list)
for a in anns:
    img2anns[a["image_id"]].append(a)

# Multi-annotation images
multi = [iid for iid, al in img2anns.items() if len(al) > 1]
print(f"\n#images with >1 annotation: {len(multi)}")
no_ann = [im["id"] for im in images if im["id"] not in img2anns]
print(f"#images with 0 annotations: {len(no_ann)}")

# Category counts (by annotation)
cat_counts = Counter(catid2name.get(a["category_id"], f"??{a['category_id']}") for a in anns)
print(f"\n=== CATEGORY COUNTS (by annotation), {len(cat_counts)} categories ===")
for name, n in cat_counts.most_common():
    print(f"{n:>10}  {name}")

# Location vs folder check
print("\n=== LOCATION vs FOLDER ===")
has_loc = sum(1 for im in images if "location" in im)
print(f"images with 'location' field: {has_loc}/{len(images)}")
# folder = dirname of file_name
loc2folders = defaultdict(set)
folder2locs = defaultdict(set)
for im in images:
    loc = im.get("location", "MISSING")
    folder = os.path.dirname(im["file_name"]).replace("\\", "/")
    loc2folders[loc].add(folder)
    folder2locs[folder].add(loc)

print(f"#distinct locations: {len(loc2folders)}")
print(f"#distinct folders:   {len(folder2locs)}")
multi_folder_locs = {l: fs for l, fs in loc2folders.items() if len(fs) > 1}
multi_loc_folders = {f: ls for f, ls in folder2locs.items() if len(ls) > 1}
print(f"locations spanning >1 folder: {len(multi_folder_locs)}")
print(f"folders spanning >1 location: {len(multi_loc_folders)}")
# Show how deep folders go vs location granularity
print("\nSample location -> folders (first 5 that differ):")
shown = 0
for l, fs in loc2folders.items():
    print(f"  loc={l!r} -> {len(fs)} folders e.g. {list(fs)[:2]}")
    shown += 1
    if shown >= 5:
        break

# file_name sample paths
print("\nSample file_names:")
for im in images[:5]:
    print("  ", im["file_name"])
