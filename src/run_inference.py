"""
Batch image classification -> MegaDetector-format output.

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

#%% Imports and constants

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

# fixed-decimal JSON: confidences as 4 decimal places, bboxes as ints
_CONF = "@@CONF@@"


#%% Support functions

def _c(x):
    """
    Wrap a confidence value in the _CONF sentinel string so it survives json.dumps as a string
    and can later be unquoted by dump_json into a bare four-decimal number.

    Args:
        x (float): a confidence value, typically in [0, 1]

    Returns:
        str: the value formatted to four decimals, prefixed with the _CONF sentinel
    """

    return f"{_CONF}{x:.4f}"


def dump_json(obj, path):
    """
    Serialize `obj` to `path` as indented JSON, then unquote the _CONF sentinel tokens so that
    confidence values appear as bare four-decimal numbers rather than quoted strings.

    Args:
        obj (dict): the object to serialize, a MegaDetector-format results dict
        path (str): output path for the JSON file
    """

    text = json.dumps(obj, indent=1, ensure_ascii=False)
    text = re.sub(rf'"{re.escape(_CONF)}(-?\d+\.\d+)"', r"\1", text)  # unquote conf tokens
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def list_images(root):
    """
    Recursively find every image file under `root` (by extension), as forward-slash relative
    paths sorted alphabetically. This sorted order is the canonical ordering used for the output
    file and for splitting work across GPU shards.

    Args:
        root (str): folder to search recursively for images

    Returns:
        list of str: image paths relative to `root`, forward-slashed and sorted alphabetically
    """

    rels = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                rels.append(rel)
    rels.sort()  # alphabetical by relative path
    return rels


class InferDataset(Dataset):
    """
    Torch Dataset that loads and preprocesses one image at a time for inference. On a per-image
    read or decode failure it returns a zero tensor and an "ok" flag of 0 instead of raising, so a
    single unreadable image is reported as a failure rather than aborting the whole run.
    """

    def __init__(self, root, rels, transform, img_size):
        """
        Initializes InferDataset.

        Args:
            root (str): folder that the `rels` paths are relative to
            rels (list of str): image paths relative to `root`
            transform (callable): preprocessing transform applied to each loaded PIL image
            img_size (int): square input size, used to build the zero tensor returned on a
                failed read
        """

        self.root = root
        self.rels = rels
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: the number of images
        """

        return len(self.rels)

    def __getitem__(self, i):
        """
        Load, convert, and transform the image at index `i`, degrading gracefully on failure.

        Args:
            i (int): index into the relative-path list

        Returns:
            tuple: (tensor, i, ok), where tensor is the transformed image (or a zero tensor on a
            read failure), i is the passed-in index, and ok is 1 on success or 0 on failure
        """

        try:
            img = Image.open(os.path.join(self.root, self.rels[i])).convert("RGB")
            return self.transform(img), i, 1
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size), i, 0


def load_model(model_path, device):
    """
    Load a stripped inference checkpoint and rebuild the timm model and its matching validation
    transform (using the banner-crop and normalization recorded in the checkpoint).

    Args:
        model_path (str): path to a *.stripped.ckpt inference checkpoint
        device (str): torch device to move the model to, e.g. "cuda" or "cpu"

    Returns:
        tuple: (model, transform, ck), where model is the eval-mode timm model on `device`,
        transform is the ValTransform to apply to each image, and ck is the loaded checkpoint dict
    """

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


#%% Inference functions

def infer(model,
          transform,
          img_size,
          root,
          rels,
          batch_size,
          workers,
          n_top,
          precision,
          device,
          progress_prefix=""):
    """
    Run the model over `rels` (relative paths), loading images through a background DataLoader and
    running whole batches on the model. A per-image read failure is recorded as a load failure; a
    whole-batch exception marks every image in that batch as an inference failure.

    Args:
        model (torch.nn.Module): the eval-mode classification model
        transform (callable): preprocessing transform applied to each image
        img_size (int): square input size, used by the dataset for failed-read placeholders
        root (str): folder that `rels` are relative to
        rels (list of str): image paths relative to `root` to run inference on
        batch_size (int): number of images per batch
        workers (int): DataLoader worker processes
        n_top (int): number of top classifications to keep per image
        precision (str): "bf16" for autocast inference or "fp32" for full precision
        device (str): torch device to run on, e.g. "cuda" or "cpu"
        progress_prefix (str, optional): string prepended to progress prints (e.g. a shard tag)

    Returns:
        list: one entry per image in `rels`, each ("OK", [(class_idx, conf), ...]) for a
        successful prediction, ("FAIL_LOAD",) for a read failure, or ("FAIL_INFER",) for a
        batch inference failure
    """

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

# ...def infer(...)


def build_output(rels, results, classes, model_basename, epoch, model_name=None):
    """
    Assemble the per-image inference results into a MegaDetector-format results dict. Each
    successful image gets a single synthetic whole-image detection with the top-N classifications
    attached; failures are written as per-image failure entries.

    Args:
        rels (list of str): image paths, in output order
        results (list): per-image entries as returned by infer()
        classes (list of str): class names, indexed by class id
        model_basename (str): basename of the model file, recorded in the output info block
        epoch (int): training epoch recorded in the output info block
        model_name (str, optional): timm model name recorded in the output info block

    Returns:
        tuple: (out, n_fail), where out is the MegaDetector-format results dict and n_fail is the
        number of images recorded as failures
    """

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
    """
    Per-GPU worker entry point: run inference over this shard's contiguous slice of the sorted
    image list and pickle the results (plus metadata) to the shard output file. The parent sets
    CUDA_VISIBLE_DEVICES so this process sees a single GPU.

    Args:
        args (argparse.Namespace): parsed CLI arguments, including the internal sharding fields
            _shard_index, _shard_count, and _shard_output
    """

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


def run_parallel(folder, model, output, batch_size, workers, classifications,
                 precision, devices, rels):
    """
    Parent-side multi-GPU driver: launch one subprocess per GPU (each pinned to one device via
    CUDA_VISIBLE_DEVICES), wait for them, and merge their pickled shard results back into a single
    ordered result set.

    Args:
        folder (str): image folder passed to each shard
        model (str): path to the stripped inference checkpoint
        output (str): output JSON path (passed through to the shard subprocesses)
        batch_size (int): per-GPU batch size
        workers (int): per-GPU DataLoader workers
        classifications (int): number of top classifications to keep per image
        precision (str): "bf16" or "fp32"
        devices (int): number of GPUs / shards to launch
        rels (list of str): the full sorted image list, used to size and order the merged results

    Returns:
        tuple: (out, n_fail) as returned by build_output over the merged results
    """

    tmpdir = tempfile.mkdtemp(prefix="csa_infer_")
    procs = []
    try:
        for i in range(devices):
            outp = os.path.join(tmpdir, f"shard_{i}.pkl")
            cmd = [sys.executable, os.path.abspath(__file__), folder, model,
                   output, "--batch-size", str(batch_size),
                   "--workers", str(workers), "--classifications",
                   str(classifications), "--precision", precision,
                   "--_shard-index", str(i), "--_shard-count", str(devices),
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


def run_inference(folder,
                  model,
                  output,
                  batch_size=32,
                  workers=8,
                  classifications=3,
                  precision="bf16",
                  devices=1):
    """
    Run classification inference over all images under `folder` and write a MegaDetector-format
    JSON to `output`. Uses multi-GPU sharding when devices > 1 and a GPU is available, otherwise
    runs in-process on a single device.

    Args:
        folder (str): image folder to search recursively
        model (str): path to a *.stripped.ckpt inference checkpoint
        output (str): output JSON path (MegaDetector format)
        batch_size (int, optional): images per batch
        workers (int, optional): DataLoader workers per GPU
        classifications (int, optional): number of top classifications to keep per image
        precision (str, optional): "bf16" (default) or "fp32"
        devices (int, optional): number of GPUs to shard across; 1 runs in-process

    Returns:
        tuple: (out, n_fail), where out is the MegaDetector-format results dict and n_fail is the
        number of failures, or None if `folder` contains no images
    """

    rels = list_images(folder)
    print(f"{len(rels):,} images under {folder}  | devices={devices} "
          f"| batch={batch_size} workers={workers}/gpu "
          f"top-{classifications} | precision={precision}", flush=True)

    if not rels:
        return None

    if devices > 1 and torch.cuda.is_available():
        out, n_fail = run_parallel(folder, model, output, batch_size, workers,
                                   classifications, precision, devices, rels)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision == "fp32":
            torch.set_float32_matmul_precision("highest")
        model_obj, transform, ck = load_model(model, device)
        n_top = min(classifications, len(ck["classes"]))
        res = infer(model_obj, transform, ck["img_size"], folder, rels,
                    batch_size, workers, n_top, precision, device)
        out, n_fail = build_output(rels, res, ck["classes"],
                                   os.path.basename(model), ck.get("epoch"),
                                   ck["model_name"])

    dump_json(out, output)
    print(f"wrote {output}  ({len(out['images']):,} images, {n_fail} failures)", flush=True)
    return out, n_fail

# ...def run_inference(...)


#%% Command-line driver

def main():
    """
    Command-line entry point: parse arguments and either run a single per-GPU shard worker (when
    the internal sharding flags are present) or drive a full inference run via run_inference.
    """

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

    run_inference(args.folder, args.model, args.output, batch_size=args.batch_size,
                  workers=args.workers, classifications=args.classifications,
                  precision=args.precision, devices=args.devices)


if __name__ == "__main__":
    main()
