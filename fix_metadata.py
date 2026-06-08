"""
Generate a corrected copy of the California Small Animals metadata JSON.

This script documents (and applies) the metadata fixes we identified. It is
intentionally explicit and defensive: it asserts that each target category still
looks the way we expect before changing it, so if the source file changes in the
future the script will fail loudly rather than silently corrupting something.

Fixes applied
-------------
1. id 230 "zebra-tailed lizard": `order` is "phrynosomatidae" (a family, not an
   order). Set `order` -> "squamata".

2. id 235 "western diamondback rattlesnake": `family` is misspelled "viperdae".
   Set `family` -> "viperidae".

3. id 239 "aspidocelis species" is a misspelled duplicate of id 158
   "aspidoscelis species" (same genus Aspidoscelis, family Teiidae; different
   wi_taxon_id). Merge: remap every annotation with category_id 239 -> 158, then
   remove category 239 from the `categories` list.

NOT changed
-----------
- id 44 "ensifera species" (labeled as the Andean Sword-billed Hummingbird on 2
  images). This is a likely annotation/auto-ID artifact, not a metadata-structure
  bug, so we leave the metadata as-is.

Input : E:\\data\\california-small-animals\\california_small_animals_with_sequences.json
Output: E:\\data\\california-small-animals\\california_small_animals_with_sequences_fixed.json
"""
import json
import os
from collections import Counter

SRC = r"E:\data\california-small-animals\california_small_animals_with_sequences.json"
DST = r"E:\data\california-small-animals\california_small_animals_with_sequences_fixed.json"

# (id, field, expected_old_value, new_value)
FIELD_FIXES = [
    (230, "name",  "zebra-tailed lizard",            None),  # sanity-check name only
    (230, "order", "phrynosomatidae",                "squamata"),
    (235, "name",  "western diamondback rattlesnake", None),
    (235, "family", "viperdae",                       "viperidae"),
]

# Merge: remap annotations from DUP_ID -> KEEP_ID, then delete DUP_ID category.
KEEP_ID = 158   # "aspidoscelis species"  (correct spelling)
DUP_ID  = 239   # "aspidocelis species"   (misspelled duplicate)


def main():
    print("Loading source JSON (1.3GB)...", flush=True)
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    cats_by_id = {c["id"]: c for c in data["categories"]}

    # --- sanity-check the merge endpoints before touching anything ---
    keep, dup = cats_by_id[KEEP_ID], cats_by_id[DUP_ID]
    assert keep["name"] == "aspidoscelis species", keep["name"]
    assert dup["name"] == "aspidocelis species", dup["name"]
    assert keep.get("genus") == dup.get("genus") == "aspidoscelis", (keep, dup)
    print(f"Merge endpoints OK: keep id {KEEP_ID} {keep['name']!r}, "
          f"drop id {DUP_ID} {dup['name']!r}")

    # --- apply scalar field fixes ---
    for cid, field, old, new in FIELD_FIXES:
        cat = cats_by_id[cid]
        cur = cat.get(field)
        assert cur == old, f"id {cid} {field}: expected {old!r}, found {cur!r}"
        if new is not None:
            cat[field] = new
            print(f"Fixed id {cid} {field}: {old!r} -> {new!r}")

    # --- remap annotations DUP_ID -> KEEP_ID ---
    remapped = 0
    for a in data["annotations"]:
        if a["category_id"] == DUP_ID:
            a["category_id"] = KEEP_ID
            remapped += 1
    print(f"Remapped {remapped} annotations from category {DUP_ID} -> {KEEP_ID}")

    # --- drop the duplicate category ---
    before = len(data["categories"])
    data["categories"] = [c for c in data["categories"] if c["id"] != DUP_ID]
    print(f"Removed duplicate category {DUP_ID}; "
          f"categories {before} -> {len(data['categories'])}")

    # --- verify no annotation still references the removed id ---
    leftover = sum(1 for a in data["annotations"] if a["category_id"] == DUP_ID)
    assert leftover == 0, f"{leftover} annotations still reference removed id {DUP_ID}"
    valid_ids = {c["id"] for c in data["categories"]}
    bad = [a["id"] for a in data["annotations"] if a["category_id"] not in valid_ids]
    assert not bad, f"{len(bad)} annotations reference unknown category ids"

    print("Writing corrected JSON...", flush=True)
    tmp = DST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, DST)
    print(f"Done -> {DST}")
    print(f"  images={len(data['images'])}  annotations={len(data['annotations'])}  "
          f"categories={len(data['categories'])}")


if __name__ == "__main__":
    main()
