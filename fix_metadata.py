"""
Verify and normalize the California Small Animals metadata JSON, in place.

This script documents (and applies) the metadata fixes we identified. The fixes are
already baked into the single canonical metadata file, so on a normal run it just
verifies they are present and rewrites the file with line breaks; it remains the
record of what those fixes were. It is intentionally explicit and defensive: it
asserts that each target category looks the way we expect, so it fails loudly rather
than silently corrupting something. It is idempotent -- each fix is applied only if
still needed, so re-running is safe -- and it always writes the output with line
breaks (indent=1) for human readability.

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

Operates in place on the single canonical metadata file:
  E:\\data\\california-small-animals\\california_small_animals_with_sequences.json
"""
import json
import os
import sys

# Single canonical metadata file; this script verifies the fixes and rewrites it in place.
SRC = r"E:\data\california-small-animals\california_small_animals_with_sequences.json"
DST = SRC

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

    # --- apply scalar field fixes (idempotent) ---
    for cid, field, old, new in FIELD_FIXES:
        cat = cats_by_id.get(cid)
        assert cat is not None, f"category {cid} not found"
        cur = cat.get(field)
        if new is None:
            assert cur == old, f"id {cid} {field}: expected {old!r}, found {cur!r}"  # name sanity check
            continue
        if cur == new:
            print(f"id {cid} {field}: already {new!r} (no change)")
        elif cur == old:
            cat[field] = new
            print(f"Fixed id {cid} {field}: {old!r} -> {new!r}")
        else:
            sys.exit(f"id {cid} {field}: expected {old!r} or {new!r}, found {cur!r}")

    # --- merge duplicate category DUP_ID -> KEEP_ID (idempotent) ---
    keep = cats_by_id.get(KEEP_ID)
    assert keep is not None and keep["name"] == "aspidoscelis species", keep
    if DUP_ID in cats_by_id:
        dup = cats_by_id[DUP_ID]
        assert dup["name"] == "aspidocelis species", dup["name"]
        assert keep.get("genus") == dup.get("genus") == "aspidoscelis", (keep, dup)
        remapped = 0
        for a in data["annotations"]:
            if a["category_id"] == DUP_ID:
                a["category_id"] = KEEP_ID
                remapped += 1
        before = len(data["categories"])
        data["categories"] = [c for c in data["categories"] if c["id"] != DUP_ID]
        print(f"Merged duplicate category {DUP_ID} -> {KEEP_ID}: remapped {remapped} "
              f"annotations; categories {before} -> {len(data['categories'])}")
    else:
        print(f"Duplicate category {DUP_ID} already merged into {KEEP_ID} (no change)")

    # --- verify no annotation still references the removed id ---
    leftover = sum(1 for a in data["annotations"] if a["category_id"] == DUP_ID)
    assert leftover == 0, f"{leftover} annotations still reference removed id {DUP_ID}"
    valid_ids = {c["id"] for c in data["categories"]}
    bad = [a["id"] for a in data["annotations"] if a["category_id"] not in valid_ids]
    assert not bad, f"{len(bad)} annotations reference unknown category ids"

    print("Writing corrected JSON...", flush=True)
    tmp = DST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)  # line breaks for readability
    os.replace(tmp, DST)
    print(f"Done -> {DST}")
    print(f"  images={len(data['images'])}  annotations={len(data['annotations'])}  "
          f"categories={len(data['categories'])}")


if __name__ == "__main__":
    main()
