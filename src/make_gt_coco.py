"""Build a master COCO Camera Traps ground-truth file for the curated training dataset.

Covers every image in the resized ``train/`` + ``val/`` tree under ``TRAIN_ROOT`` and emits a
single COCO CT JSON whose annotations carry the folder-derived class label. This is the reusable
ground-truth artifact for downstream analysis (confusion matrices, label cleanup, MegaDetector
``analyze_classification_results``, ...).

The image list and labels come from the locked ``split.parquet`` (the curated subset), filtered to
the images that actually exist on disk (so the file lists only reviewable images -- one source
image failed to copy). **Per-image sequence information is preserved** (``seq_id``,
``seq_num_frames``, ``frame_num``) along with ``datetime``; ``seq_num_frames`` and ``datetime`` are
joined from the canonical metadata (the path-config's ``METADATA_FILE``), the rest come from
``split.parquet``.

Output: ``<TRAIN_ROOT>/california-small-animals-training.json`` (written with ``indent=1``).

Conventions:
  - ``file_name`` / image ``id``: ``"<split>/<class>/<file>.jpg"`` relative to TRAIN_ROOT, forward
    slashes. (Prediction files store paths relative to each split root, i.e. without the leading
    ``"<split>/"`` -- prefix the split to match.)
  - ``source_image_id``: the original dataset image id (GUID).
  - ``category_id``: index into ``label_map.CLASS_ORDER`` (0-based; matches the classifier's class
    ids and the prediction files' ``classification_categories`` keys).
  - ``image["split"]``: ``"train"`` or ``"val"``.
One annotation per image (single label).
"""
import os
import sys
import json
import argparse
import datetime

import pandas as pd

from label_map import CLASS_ORDER
from path_config import load_path_config

SPLITS = ["train", "val"]


def existing_images(train_root):
    """Set of '<split>/<class>/<file>.jpg' rel-paths actually present on disk."""
    existing = set()
    for split in SPLITS:
        split_dir = os.path.join(train_root, split)
        if not os.path.isdir(split_dir):
            sys.exit(f"Missing split dir: {split_dir}")
        for cls in os.scandir(split_dir):
            if not cls.is_dir():
                continue
            with os.scandir(cls.path) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        existing.add(f"{split}/{cls.name}/{e.name}")
    return existing


def seq_extra_from_meta(meta_path):
    """image_id -> (seq_num_frames, datetime) from the canonical metadata."""
    print(f"loading metadata for sequence info: {meta_path}", flush=True)
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    extra = {im["id"]: (im.get("seq_num_frames"), im.get("datetime")) for im in data["images"]}
    del data
    return extra


def build(train_root, out_base, meta):
    name_to_id = {name: i for i, name in enumerate(CLASS_ORDER)}
    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_ORDER)]

    existing = existing_images(train_root)
    print(f"images on disk: {len(existing):,}", flush=True)
    extra = seq_extra_from_meta(meta)
    df = pd.read_parquet(os.path.join(out_base, "split.parquet"))
    print(f"split.parquet rows: {len(df):,}", flush=True)

    images = []
    annotations = []
    per_split = {s: 0 for s in SPLITS}
    skipped_missing_file = 0
    missing_extra = 0

    for r in df.itertuples(index=False):
        rel = r.dest_rel
        if rel not in existing:
            skipped_missing_file += 1
            continue
        snf, dt = extra.get(r.image_id, (None, None))
        if snf is None:
            missing_extra += 1
        im = {
            "id": rel,
            "file_name": rel,
            "source_image_id": r.image_id,
            "location": r.location,
            "datetime": dt,
            "width": int(r.width),
            "height": int(r.height),
            "seq_id": r.seq_id,
            "seq_num_frames": (int(snf) if snf is not None else None),
            "frame_num": int(r.frame_num),
            "split": r.split,
        }
        images.append(im)
        annotations.append({"id": rel + "_ann", "image_id": rel, "category_id": name_to_id[r.target_class]})
        per_split[r.split] += 1

    for s in SPLITS:
        print(f"{s}: {per_split[s]:,} images", flush=True)
    print(f"skipped (in split.parquet, not on disk): {skipped_missing_file:,}", flush=True)
    if missing_extra:
        print(f"WARNING: {missing_extra:,} images missing seq_num_frames/datetime in metadata", flush=True)

    coco = {
        "info": {
            "description": "California Small Animals -- curated training dataset ground truth "
                           "(folder-derived labels for the resized train/val tree; sequence "
                           "information preserved).",
            "version": "1.1",
            "date_created": datetime.datetime.now().isoformat(timespec="seconds"),
            "category_id_scheme": "index into label_map.CLASS_ORDER (0-based)",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path-config", required=True,
                    help="JSON file of machine paths (TRAIN_ROOT, OUTPUT_ROOT, METADATA_FILE)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <TRAIN_ROOT>/california-small-animals-training.json)")
    ap.add_argument("--force", action="store_true", help="overwrite if it already exists")
    args = ap.parse_args()

    cfg = load_path_config(args.path_config)
    out_path = args.out or os.path.join(cfg.TRAIN_ROOT, "california-small-animals-training.json")

    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} already exists; pass --force to overwrite")

    coco = build(cfg.TRAIN_ROOT, cfg.OUTPUT_ROOT, cfg.METADATA_FILE)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=1)  # line breaks for readability
    os.replace(tmp, out_path)
    sz = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({sz:.1f} MB): {len(coco['images'])} images, "
          f"{len(coco['categories'])} categories", flush=True)


if __name__ == "__main__":
    main()
