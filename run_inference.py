"""Batch image classification -> MegaDetector-format output.

This is a classification-only model, so each image gets a single synthetic
whole-image "detection" (category "object", bbox [0,0,1,1]); our class predictions
are attached to that detection as the top-N [category_id, conf] classifications,
sorted high->low. Operates recursively on a folder of images.

Loading/preprocessing happens in background DataLoader workers (one image at a
time); the main thread pulls whole batches and runs the model. Per-image read
failures and whole-batch inference failures are reported per the format spec.

Usage:
  python run_inference.py <image-folder> <model.stripped.ckpt> <output.json>
      [--batch-size 32] [--workers 8] [--classifications 3]
"""
import argparse
import contextlib
import datetime
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transforms import ValTransform

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORMAT_VERSION = "1.4"
FAIL_LOAD = "Failure image access"
FAIL_INFER = "Failure inference"

# --- fixed-decimal JSON: confidences as N decimal places, bboxes as ints ---
_CONF = "@@CONF@@"


def _c(x):
    """Wrap a confidence so it serializes with exactly 4 decimals (see dump_json)."""
    return f"{_CONF}{x:.4f}"


def dump_json(obj, path):
    text = json.dumps(obj, indent=1, ensure_ascii=False)
    text = re.sub(rf'"{re.escape(_CONF)}(-?\d+\.\d+)"', r"\1", text)  # unquote conf tokens
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def list_images(root):
    rels = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                rels.append(rel)
    rels.sort()  # alphabetical by relative path
    return rels


class InferDataset(Dataset):
    def __init__(self, root, rels, transform, img_size):
        self.root = root
        self.rels = rels
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(self.root, self.rels[i])).convert("RGB")
            return self.transform(img), i, 1
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size), i, 0


def load_model(model_path, device):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    import timm
    model = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=ck["num_classes"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    transform = ValTransform(img_size=ck["img_size"],
                             banner_top=ck["banner_crop"]["top"],
                             banner_bot=ck["banner_crop"]["bottom"],
                             crop_banner_flag=True,
                             mean=tuple(ck["norm_mean"]), std=tuple(ck["norm_std"]))
    return model, transform, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="image folder (searched recursively)")
    ap.add_argument("model", help="path to a *.stripped.ckpt inference checkpoint")
    ap.add_argument("output", help="output .json path (MegaDetector format)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--classifications", type=int, default=3,
                    help="top-N classifications per image (default 3)")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 (default, fast, matches training) or fp32 "
                         "(slower, ~deterministic across batch sizes)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp32":
        torch.set_float32_matmul_precision("highest")  # true fp32, deterministic
    model, transform, ck = load_model(args.model, device)
    classes = ck["classes"]
    n_top = min(args.classifications, len(classes))

    rels = list_images(args.folder)
    print(f"{len(rels):,} images under {args.folder}  | device={device} "
          f"| batch={args.batch_size} workers={args.workers} top-{n_top}")
    if not rels:
        return

    ds = InferDataset(args.folder, rels, transform, ck["img_size"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"),
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None)

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (device == "cuda" and args.precision == "bf16")
           else contextlib.nullcontext())
    results = [None] * len(rels)   # idx -> ("OK", pairs) | ("FAIL_LOAD",) | ("FAIL_INFER",)
    done = 0
    for tensors, idxs, oks in loader:
        idxs = idxs.tolist()
        oks = oks.tolist()
        try:
            with torch.no_grad(), amp:
                logits = model(tensors.to(device, non_blocking=True))
            probs = torch.softmax(logits.float(), dim=1).cpu()
            topv, topi = probs.topk(n_top, dim=1)
            batch_failed = False
        except Exception as e:
            batch_failed = True
            print(f"  batch inference failure ({type(e).__name__}: {e}) -> "
                  f"{len(idxs)} images marked failed")

        for j, (idx, ok) in enumerate(zip(idxs, oks)):
            if batch_failed:
                results[idx] = ("FAIL_INFER",)
            elif not ok:
                results[idx] = ("FAIL_LOAD",)
            else:
                pairs = [(int(ci), float(cv))
                         for cv, ci in zip(topv[j].tolist(), topi[j].tolist())]
                results[idx] = ("OK", pairs)
        done += len(idxs)
        if done % (args.batch_size * 50) < args.batch_size:
            print(f"  {done:,}/{len(rels):,}", flush=True)

    images = []
    n_fail = 0
    for idx, rel in enumerate(rels):
        r = results[idx]
        if r[0] == "OK":
            classifications = [[str(ci), _c(cv)] for ci, cv in r[1]]
            images.append({"file": rel, "detections": [
                {"category": "1", "conf": _c(1.0), "bbox": [0, 0, 1, 1],
                 "classifications": classifications}]})
        else:
            n_fail += 1
            images.append({"file": rel,
                           "failure": FAIL_LOAD if r[0] == "FAIL_LOAD" else FAIL_INFER})

    out = {
        "info": {
            "classifier": os.path.basename(args.model),
            "classifier_metadata": {"model_name": ck["model_name"],
                                    "epoch": ck.get("epoch")},
            "detector": "synthetic whole-image box (classification-only model)",
            "classification_completion_time":
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format_version": FORMAT_VERSION,
        },
        "detection_categories": {"1": "object"},
        "classification_categories": {str(i): c for i, c in enumerate(classes)},
        "images": images,
    }
    dump_json(out, args.output)
    print(f"wrote {args.output}  ({len(images):,} images, {n_fail} failures)")


if __name__ == "__main__":
    main()
