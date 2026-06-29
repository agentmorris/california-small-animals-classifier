"""Strip a Lightning training checkpoint into a compact, self-contained inference
checkpoint.

Lightning checkpoints (~3.5 GB here) carry optimizer state, LR schedulers, callback
state, etc. For inference we only need the model weights. This writes a
`<name>.stripped.ckpt` next to each original (originals left intact) containing the
timm model weights plus everything needed to run inference standalone (class list,
model name, input size, normalization, banner-crop preprocessing).

Usage:
  python strip_checkpoint.py <file-or-dir> [--half]
    <file-or-dir>  a .ckpt file, or a folder (all *.ckpt except *.stripped.ckpt)
    --half         store weights as float16 (smaller; default keeps float32)
"""
import argparse
import glob
import os

import torch
import timm

from label_map import CLASS_ORDER
from transforms import IMAGENET_MEAN  # noqa: F401  (kept for reference)

FORMAT = "csa-classifier-inference-v1"
# Banner-crop fractions used by the training val transform (see transforms.py).
BANNER_TOP, BANNER_BOT = 0.03, 0.035


def resolve_data_cfg(model_name):
    m = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_ORDER))
    dc = timm.data.resolve_model_data_config(m)
    del m
    return int(dc["input_size"][-1]), list(map(float, dc["mean"])), list(map(float, dc["std"]))


def strip_one(src, data_cfg_cache, half=False, dst=None):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_name = hp.get("model_name")
    num_classes = hp.get("num_classes", len(CLASS_ORDER))
    if model_name is None:
        raise ValueError(f"{src}: no model_name in hyper_parameters")

    if model_name not in data_cfg_cache:
        data_cfg_cache[model_name] = resolve_data_cfg(model_name)
    img_size, mean, std = data_cfg_cache[model_name]

    # keep only the timm backbone weights (drop optimizer, metrics, cls_weights, ...)
    sd = ckpt["state_dict"]
    model_sd = {k[len("model."):]: (v.half() if half else v)
                for k, v in sd.items() if k.startswith("model.")}

    out = {
        "format": FORMAT,
        "model_name": model_name,
        "num_classes": num_classes,
        "classes": list(CLASS_ORDER),
        "img_size": img_size,
        "norm_mean": mean,
        "norm_std": std,
        "banner_crop": {"top": BANNER_TOP, "bottom": BANNER_BOT},
        "preprocessing": "crop banner -> squash to img_size -> normalize (matches training val)",
        "weights_dtype": "float16" if half else "float32",
        "source_checkpoint": os.path.basename(src),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "state_dict": model_sd,
    }
    if dst is None:
        dst = src[:-len(".ckpt")] + ".stripped.ckpt" if src.endswith(".ckpt") else src + ".stripped.ckpt"
    torch.save(out, dst)
    return dst, os.path.getsize(src), os.path.getsize(dst), len(model_sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a .ckpt file or a folder of checkpoints")
    ap.add_argument("--half", action="store_true", help="store weights as float16")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(f for f in glob.glob(os.path.join(args.path, "*.ckpt"))
                       if not f.endswith(".stripped.ckpt"))
    else:
        files = [args.path]
    if not files:
        print("no checkpoints found")
        return

    cache = {}
    for f in files:
        dst, s0, s1, n = strip_one(f, cache, half=args.half)
        print(f"{os.path.basename(f)} -> {os.path.basename(dst)}  "
              f"{s0/1e9:.2f} GB -> {s1/1e9:.2f} GB  ({n} weight tensors)")


if __name__ == "__main__":
    main()
