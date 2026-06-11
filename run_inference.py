"""Batch image classification -> MegaDetector-format output.

This is a classification-only model, so each image gets a single synthetic
whole-image "detection" (category "object", bbox [0,0,1,1]); our class predictions
are attached to that detection as the top-N [category_id, conf] classifications,
sorted high->low. Operates recursively on a folder of images.

Loading/preprocessing happens in background DataLoader workers (one image at a
time); the main thread pulls whole batches and runs the model. Per-image read
failures and whole-batch inference failures are reported per the format spec.

Multi-GPU (`--devices N`): the sorted file list is split into N contiguous shards,
each run as its own subprocess pinned to one GPU; results are merged in order into
a single output file (identical to the 1-GPU result).

Usage:
  python run_inference.py <image-folder> <model.stripped.ckpt> <output.json>
      [--batch-size 32] [--workers 8] [--classifications 3]
      [--precision bf16|fp32] [--devices 2]
"""
import argparse
import contextlib
import datetime
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transforms import ValTransform

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORMAT_VERSION = "1.4"
FAIL_LOAD = "Failure image access"
FAIL_INFER = "Failure inference"

# --- fixed-decimal JSON: confidences as 4 decimal places, bboxes as ints ---
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


def infer(model, transform, img_size, root, rels, batch_size, workers, n_top,
          precision, device, progress_prefix=""):
    """Run the model over `rels` (relative paths). Returns a list aligned to rels:
    each entry is ("OK", [(class_idx, conf), ...]) | ("FAIL_LOAD",) | ("FAIL_INFER",)."""
    ds = InferDataset(root, rels, transform, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device == "cuda"),
                        persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (device == "cuda" and precision == "bf16") else contextlib.nullcontext())
    results = [None] * len(rels)
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
            print(f"{progress_prefix}batch inference failure ({type(e).__name__}: {e}) "
                  f"-> {len(idxs)} images marked failed")
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
        if done % (batch_size * 50) < batch_size:
            print(f"{progress_prefix}{done:,}/{len(rels):,}", flush=True)
    return results


def build_output(rels, results, classes, model_basename, epoch, model_name=None):
    images = []
    n_fail = 0
    for idx, rel in enumerate(rels):
        r = results[idx]
        if r is not None and r[0] == "OK":
            classifications = [[str(ci), _c(cv)] for ci, cv in r[1]]
            images.append({"file": rel, "detections": [
                {"category": "1", "conf": _c(1.0), "bbox": [0, 0, 1, 1],
                 "classifications": classifications}]})
        else:
            n_fail += 1
            kind = r[0] if r is not None else "FAIL_INFER"
            images.append({"file": rel,
                           "failure": FAIL_LOAD if kind == "FAIL_LOAD" else FAIL_INFER})
    out = {
        "info": {
            "classifier": model_basename,
            "classifier_metadata": {"model_name": model_name, "epoch": epoch},
            "detector": "synthetic whole-image box (classification-only model)",
            "classification_completion_time":
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format_version": FORMAT_VERSION,
        },
        "detection_categories": {"1": "object"},
        "classification_categories": {str(i): c for i, c in enumerate(classes)},
        "images": images,
    }
    return out, n_fail


def run_shard(args):
    """Worker: process this shard's contiguous slice and pickle its results.
    CUDA_VISIBLE_DEVICES (set by the parent) restricts us to a single GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp32":
        torch.set_float32_matmul_precision("highest")
    model, transform, ck = load_model(args.model, device)
    rels = list_images(args.folder)
    per = math.ceil(len(rels) / args._shard_count)
    start = args._shard_index * per
    sub = rels[start:min(len(rels), start + per)]
    n_top = min(args.classifications, len(ck["classes"]))
    res = infer(model, transform, ck["img_size"], args.folder, sub, args.batch_size,
                args.workers, n_top, args.precision, device,
                progress_prefix=f"[gpu{args._shard_index}] ")
    with open(args._shard_output, "wb") as f:
        pickle.dump({"start": start, "results": res, "classes": ck["classes"],
                     "model_basename": os.path.basename(args.model),
                     "model_name": ck["model_name"], "epoch": ck.get("epoch")}, f)


def run_parallel(args, rels):
    """Parent: launch one subprocess per GPU, merge their results in order."""
    tmpdir = tempfile.mkdtemp(prefix="csa_infer_")
    procs = []
    try:
        for i in range(args.devices):
            outp = os.path.join(tmpdir, f"shard_{i}.pkl")
            cmd = [sys.executable, os.path.abspath(__file__), args.folder, args.model,
                   args.output, "--batch-size", str(args.batch_size),
                   "--workers", str(args.workers), "--classifications",
                   str(args.classifications), "--precision", args.precision,
                   "--_shard-index", str(i), "--_shard-count", str(args.devices),
                   "--_shard-output", outp]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            procs.append((subprocess.Popen(cmd, env=env), outp))
        full = [None] * len(rels)
        meta = None
        for p, outp in procs:
            if p.wait() != 0:
                raise RuntimeError(f"a GPU shard exited with code {p.returncode}")
            with open(outp, "rb") as f:
                d = pickle.load(f)
            meta = d
            for k, r in enumerate(d["results"]):
                full[d["start"] + k] = r
        return build_output(rels, full, meta["classes"], meta["model_basename"],
                            meta["epoch"], meta["model_name"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="image folder (searched recursively)")
    ap.add_argument("model", help="path to a *.stripped.ckpt inference checkpoint")
    ap.add_argument("output", help="output .json path (MegaDetector format)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8, help="loader workers PER GPU")
    ap.add_argument("--classifications", type=int, default=3,
                    help="top-N classifications per image (default 3)")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 (default, fast, matches training) or fp32 "
                         "(slower, ~deterministic across batch sizes)")
    ap.add_argument("--devices", type=int, default=1,
                    help="number of GPUs to shard across (default 1)")
    # internal (per-GPU subprocess) flags
    ap.add_argument("--_shard-index", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-count", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-output", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._shard_index is not None:        # we are a per-GPU worker
        run_shard(args)
        return

    rels = list_images(args.folder)
    print(f"{len(rels):,} images under {args.folder}  | devices={args.devices} "
          f"| batch={args.batch_size} workers={args.workers}/gpu "
          f"top-{min(args.classifications, 999)} | precision={args.precision}")
    if not rels:
        return

    if args.devices > 1 and torch.cuda.is_available():
        out, n_fail = run_parallel(args, rels)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.precision == "fp32":
            torch.set_float32_matmul_precision("highest")
        model, transform, ck = load_model(args.model, device)
        n_top = min(args.classifications, len(ck["classes"]))
        res = infer(model, transform, ck["img_size"], args.folder, rels,
                    args.batch_size, args.workers, n_top, args.precision, device)
        out, n_fail = build_output(rels, res, ck["classes"],
                                   os.path.basename(args.model), ck.get("epoch"),
                                   ck["model_name"])

    dump_json(out, args.output)
    print(f"wrote {args.output}  ({len(out['images']):,} images, {n_fail} failures)")


if __name__ == "__main__":
    main()
