"""Generate the val-only COCO Camera Traps file (val_cct.json) from the master training GT.

Same images/labels as the 'val' split of the master GT, but with **val-relative** paths (no
leading 'val/'), so it matches the val prediction files and MegaDetector's
analyze_classification_results. Per-image sequence information is preserved. Written indent=1.

Output: <TRAIN_ROOT>/val/val_cct.json
"""
import os
import sys
import json
import argparse

from label_map import TRAIN_ROOT

MASTER = os.path.join(TRAIN_ROOT, "california-small-animals-training.json")
OUT = os.path.join(TRAIN_ROOT, "val", "val_cct.json")
PFX = "val/"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not os.path.isfile(args.master):
        sys.exit(f"master GT not found: {args.master} (run make_gt_coco.py first)")

    with open(args.master, encoding="utf-8") as f:
        gt = json.load(f)
    cat_of = {a["image_id"]: a["category_id"] for a in gt["annotations"]}

    images, annotations = [], []
    for im in gt["images"]:
        if im.get("split") != "val":
            continue
        rid = im["id"]
        assert rid.startswith(PFX), f"val image id without '{PFX}' prefix: {rid}"
        rel = rid[len(PFX):]                       # "<class>/<file>.jpg" (val-relative)
        images.append({
            "id": rel,
            "file_name": rel,
            "source_image_id": im.get("source_image_id"),
            "location": im.get("location"),
            "datetime": im.get("datetime"),
            "width": im.get("width"),
            "height": im.get("height"),
            "seq_id": im.get("seq_id"),
            "seq_num_frames": im.get("seq_num_frames"),
            "frame_num": im.get("frame_num"),
        })
        annotations.append({"id": rel + "_ann", "image_id": rel, "category_id": cat_of[rid]})

    out = {
        "info": {
            "description": "California Small Animals -- val-only ground truth (val-relative "
                           "paths; sequence info preserved).",
            "derived_from": os.path.basename(args.master),
        },
        "categories": gt["categories"],          # same CLASS_ORDER-indexed scheme as the master GT
        "images": images,
        "annotations": annotations,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {len(images):,} val images, {len(out['categories'])} categories")


if __name__ == "__main__":
    main()
