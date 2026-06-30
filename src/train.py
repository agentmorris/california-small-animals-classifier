"""Train the first-pass California Small Animals classifier (eva02_large @ 448, PTL).

Reads split.parquet, builds train/val datasets from the F: folder tree, fine-tunes
a timm backbone with the agreed augmentation, logs micro + macro accuracy, and
checkpoints every epoch.

Examples:
  python train.py --devices 1 --benchmark-steps 60     # quick throughput probe
  python train.py --devices 2                           # full run, both GPUs (DDP)
"""
import argparse
import datetime
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch._dynamo
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (ModelCheckpoint, LearningRateMonitor,
                                         EarlyStopping)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from PIL import Image
import timm
from timm.optim import param_groups_layer_decay

from label_map import CLASS_ORDER
from path_config import load_path_config, load_excluded_guids
from transforms import TrainTransform, ValTransform

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


class Throughput(L.Callback):
    """Report steady-state images/sec (skips warmup steps)."""
    def __init__(self, batch_size, warmup=8):
        self.batch_size = batch_size
        self.warmup = warmup
        self.n = 0
        self.t0 = None

    def on_train_batch_end(self, trainer, pl_module, *a):
        self.n += 1
        if self.n == self.warmup:
            torch.cuda.synchronize()
            self.t0 = time.time()

    def on_train_end(self, trainer, pl_module):
        if self.t0 is None:
            return
        torch.cuda.synchronize()
        dt = time.time() - self.t0
        steps = self.n - self.warmup
        if steps <= 0 or dt <= 0:
            return
        world = trainer.world_size
        local_it = steps / dt
        gbs = self.batch_size * world
        if trainer.is_global_zero:
            print(f"\n[THROUGHPUT] world={world} local={local_it:.2f} it/s  "
                  f"global={local_it*gbs:.0f} img/s  "
                  f"(~{1_057_239/(local_it*gbs)/60:.1f} min/epoch over 1.06M imgs)",
                  flush=True)


class IntermediateCheckpoint(L.Callback):
    """Write `per_epoch` EXTRA weights-only checkpoints spread through each epoch.

    Fires at i/(per_epoch+1) of the epoch for i in 1..per_epoch (per_epoch=1 -> halfway;
    per_epoch=2 -> 1/3 and 2/3; per_epoch=8 -> 1/9..8/9). Targets are recomputed each epoch (no
    cross-epoch drift) and never land on the epoch boundary, so these are purely additive to the
    usual full epoch-end checkpoints. Saved weights-only (~1.2 GB, no optimizer) for cheap offline
    validation later; loadable by strip_checkpoint.py / run_inference.py.
    """
    def __init__(self, dirpath, run_name, per_epoch):
        self.dirpath = dirpath
        self.run_name = run_name
        self.per_epoch = per_epoch
        self._targets = set()

    def on_train_epoch_start(self, trainer, pl_module):
        n = trainer.num_training_batches  # train batches this epoch (per rank)
        self._targets = {max(1, round(n * i / (self.per_epoch + 1)))
                         for i in range(1, self.per_epoch + 1)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) in self._targets:
            fn = f"{self.run_name}-e{trainer.current_epoch:02d}-s{trainer.global_step:06d}.ckpt"
            # called on all ranks at the same batch -> save_checkpoint coordinates the DDP gather
            trainer.save_checkpoint(os.path.join(self.dirpath, fn), weights_only=True)


class CSADataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # robust to missing/corrupt files (e.g. the known 0-byte source image):
        # fall back to the next index rather than crashing the run.
        for off in range(8):
            j = (i + off) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
                return self.transform(img), int(self.labels[j])
            except Exception:
                continue
        raise RuntimeError(f"could not load image near index {i}")


def load_frames(split_path, train_root, exclude_files, apply_exclude=True):
    df = pd.read_parquet(split_path, columns=["image_id", "dest_rel", "target_class", "split"])
    if apply_exclude:
        excluded = load_excluded_guids(exclude_files)
        if excluded:
            before = len(df)
            df = df[~df.image_id.isin(excluded)]
            print(f"manual-review: dropped {before - len(df):,} excluded images")
    df["label"] = df["target_class"].astype(str).map(CLASS_TO_IDX).astype(int)
    df["path"] = [os.path.join(train_root, d.replace("/", os.sep)) for d in df.dest_rel]
    return df[df.split == "train"], df[df.split == "val"]


def class_weights(train_labels, scheme="sqrt", cap=10.0):
    counts = np.bincount(train_labels, minlength=len(CLASS_ORDER)).astype(np.float64)
    counts = np.clip(counts, 1, None)
    freq = counts / counts.sum()
    if scheme == "none":
        w = np.ones_like(freq)
    elif scheme == "sqrt":
        w = freq ** -0.5
    elif scheme == "inv":
        w = freq ** -1.0
    else:
        raise ValueError(scheme)
    w = w / w.mean()
    w = np.clip(w, None, cap)
    return torch.tensor(w, dtype=torch.float32), counts


class Classifier(L.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay, label_smoothing,
                 warmup_epochs, max_epochs, cls_weights, grad_ckpt=True, compile=True,
                 freeze_backbone=False, layer_decay=1.0, warmup_steps=0):
        super().__init__()
        self.save_hyperparameters(ignore=["cls_weights"])
        self.model = timm.create_model(model_name, pretrained=True,
                                       num_classes=num_classes)
        if freeze_backbone:
            # linear probe: freeze the whole backbone, train only the classifier head
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in self.model.get_classifier().parameters():
                p.requires_grad_(True)
            grad_ckpt = False  # no backbone backward -> nothing to checkpoint
        if grad_ckpt and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
        # Compiled wrapper kept in a plain list so it is NOT registered as a
        # submodule -> state_dict stays clean (self.model.* only), shares params.
        self._compiled = [torch.compile(self.model)] if compile else None
        self.register_buffer("cls_weights", cls_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.cls_weights,
                                             label_smoothing=label_smoothing)
        self.train_acc = MulticlassAccuracy(num_classes, average="micro")
        self.val_micro = MulticlassAccuracy(num_classes, average="micro")
        self.val_macro = MulticlassAccuracy(num_classes, average="macro")

    def forward(self, x):
        return self._compiled[0](x) if self._compiled is not None else self.model(x)

    def train(self, mode=True):
        # keep a frozen backbone in eval mode (deterministic features: no drop_path/dropout)
        super().train(mode)
        if mode and self.hparams.freeze_backbone:
            self.model.eval()
        return self

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_micro(logits, y)
        self.val_macro(logits, y)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc_micro", self.val_micro, prog_bar=True)
        self.log("val/acc_macro", self.val_macro, prog_bar=True)

    def configure_optimizers(self):
        base_lr, wd = self.hparams.lr, self.hparams.weight_decay

        if self.hparams.freeze_backbone:
            # linear probe: only the (unfrozen) head params go to the optimizer
            params = [p for p in self.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
        elif self.hparams.layer_decay and self.hparams.layer_decay < 1.0:
            # layer-wise LR decay: bake per-layer lr into each group so the standard
            # warmup/cosine schedulers (which scale every group's initial_lr by the same
            # factor) preserve the per-layer ratios.
            groups = param_groups_layer_decay(
                self.model, weight_decay=wd, layer_decay=self.hparams.layer_decay,
                no_weight_decay_list=self.model.no_weight_decay())
            for g in groups:
                g["lr"] = base_lr * g.get("lr_scale", 1.0)
            opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=wd)

        L = torch.optim.lr_scheduler
        if self.hparams.warmup_steps and self.hparams.warmup_steps > 0:
            total = max(2, int(self.trainer.estimated_stepping_batches))
            wu = min(self.hparams.warmup_steps, total - 1)
            warmup = L.LinearLR(opt, start_factor=0.01, total_iters=wu)
            cosine = L.CosineAnnealingLR(opt, T_max=max(1, total - wu))
            sched = L.SequentialLR(opt, [warmup, cosine], milestones=[wu])
            interval = "step"
        else:
            warmup = L.LinearLR(opt, start_factor=0.01, total_iters=max(1, self.hparams.warmup_epochs))
            cosine = L.CosineAnnealingLR(opt, T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            sched = L.SequentialLR(opt, [warmup, cosine], milestones=[self.hparams.warmup_epochs])
            interval = "epoch"
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": interval}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-config", required=True,
                    help="JSON file of machine-specific paths (OUTPUT_ROOT, TRAIN_ROOT, "
                         "EXCLUDE_FILES, ...); see path_config.example.json")
    ap.add_argument("--model", default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=24, help="per GPU")
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--weight-scheme", default="sqrt", choices=["none", "sqrt", "inv"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile (compile is on by default)")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="if >0, run only this many train steps (throughput probe)")
    ap.add_argument("--run-name", required=True,
                    help="run folder name under <OUTPUT_ROOT>/runs/")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'last' to resume from checkpoints/last.ckpt")
    ap.add_argument("--patience", type=int, default=0,
                    help="if >0, enable early stopping on val/acc_macro (mode max) "
                         "with this patience in epochs")
    ap.add_argument("--no-exclude", action="store_true",
                    help="train on every image in the split, ignoring manual-review exclusions")
    ap.add_argument("--checkpoint-folder", default=None,
                    help="write checkpoints to <folder>/<run-name>/ during training instead of the "
                         "run folder (use a fast WSL-local/ext4 path to keep the big per-epoch "
                         "writes off the 9p run folder); on a clean finish they are moved into "
                         "runs/<run-name>/checkpoints/. On a crash/cancel, move them manually.")
    ap.add_argument("--intermediate-checkpoints-per-epoch", type=int, default=0,
                    help="if >0, also write this many EXTRA weights-only checkpoints spread through "
                         "each epoch, at i/(N+1) of the way (N=1 -> halfway; N=2 -> 1/3 and 2/3; "
                         "N=8 -> 1/9..8/9). The usual full epoch-end checkpoints are still written.")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="linear-probe mode: freeze the backbone (kept in eval), train only the "
                         "classifier head. Pair with a higher --lr (e.g. 1e-3).")
    ap.add_argument("--layer-decay", type=float, default=1.0,
                    help="layer-wise LR decay for full fine-tuning (e.g. 0.75): early blocks get "
                         "exponentially smaller LR. 1.0 disables it; ignored with --freeze-backbone.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="if >0, use a step-based LR schedule (this many warmup steps, then cosine "
                         "over the remaining steps). Recommended; otherwise warmup is the legacy "
                         "per-epoch --warmup-epochs.")
    args = ap.parse_args()

    cfg = load_path_config(args.path_config)
    split_path = os.path.join(cfg.OUTPUT_ROOT, "split.parquet")
    runs_dir = os.path.join(cfg.OUTPUT_ROOT, "runs")   # each run lives in runs/<run-name>/

    # Per-run output folder: runs/<run-name>/ holds checkpoints/, metrics.csv,
    # hparams.yaml, and (via the launcher) the train log.
    run_name = args.run_name
    run_dir = os.path.join(runs_dir, run_name)
    final_ckpt_dir = os.path.join(run_dir, "checkpoints")  # durable home, in the run folder
    # Optionally write checkpoints to a separate (e.g. fast WSL-local) folder during the run so the
    # big per-epoch writes don't hit the slow/9p run folder; moved into the run folder at the end of
    # a clean run (see after trainer.fit). metrics.csv / hparams.yaml / log stay in the run folder.
    ckpt_dir = (os.path.join(args.checkpoint_folder, run_name)
                if args.checkpoint_folder else final_ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Lightning's CSVLogger truncates metrics.csv when it re-inits on a resumed run (it does not
    # read prior rows), so preserve the prior segment first. Each resume leaves a timestamped
    # metrics.<ts>.csv beside the fresh metrics.csv, so the full per-epoch history spans all
    # metrics*.csv in the run folder. rank-0 only (this runs before DDP spawns; guard on LOCAL_RANK).
    if args.resume and os.environ.get("LOCAL_RANK", "0") == "0":
        prior_metrics = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(prior_metrics):
            ts = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            backup = os.path.join(run_dir, f"metrics.{ts}.csv")
            shutil.copy2(prior_metrics, backup)
            print(f"resume: backed up metrics.csv -> {os.path.basename(backup)}", flush=True)

    torch.set_float32_matmul_precision("high")
    # torch.compile's DDPOptimizer (graph bucket-splitting) chokes on EVA's
    # rope/SDPA subgraph; disable it so compile + DDP coexist.
    if not args.no_compile and args.devices > 1:
        torch._dynamo.config.optimize_ddp = False

    train_df, val_df = load_frames(split_path, cfg.TRAIN_ROOT, cfg.EXCLUDE_FILES,
                                   apply_exclude=not args.no_exclude)

    # resolve normalization from the actual model cfg
    tmp = timm.create_model(args.model, pretrained=False, num_classes=len(CLASS_ORDER))
    dc = timm.data.resolve_model_data_config(tmp)
    mean, std = dc["mean"], dc["std"]
    del tmp
    print(f"norm mean={mean} std={std}")

    tt = TrainTransform(args.img_size, mean=mean, std=std)
    vt = ValTransform(args.img_size, crop_banner_flag=True, mean=mean, std=std)

    train_ds = CSADataset(train_df.path.tolist(), train_df.label.to_numpy(), tt)
    val_ds = CSADataset(val_df.path.tolist(), val_df.label.to_numpy(), vt)
    print(f"train={len(train_ds):,}  val={len(val_ds):,}")

    cls_w, counts = class_weights(train_df.label.to_numpy(), args.weight_scheme)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0,
                              prefetch_factor=4 if args.workers else None, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers else None)

    model = Classifier(args.model, len(CLASS_ORDER), args.lr, args.weight_decay,
                       args.label_smoothing, args.warmup_epochs, args.epochs,
                       cls_w, grad_ckpt=not args.no_grad_ckpt,
                       compile=not args.no_compile,
                       freeze_backbone=args.freeze_backbone,
                       layer_decay=args.layer_decay,
                       warmup_steps=args.warmup_steps)

    ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename=run_name + "-{epoch:02d}",
                           every_n_epochs=1, save_top_k=-1, save_last=True,
                           auto_insert_metric_name=False)
    lrmon = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lrmon, Throughput(args.batch_size)]
    if not args.benchmark_steps:
        callbacks.append(ckpt)
        if args.intermediate_checkpoints_per_epoch > 0:
            callbacks.append(IntermediateCheckpoint(
                ckpt_dir, run_name, args.intermediate_checkpoints_per_epoch))
            print(f"intermediate checkpoints: {args.intermediate_checkpoints_per_epoch}/epoch "
                  f"(extra, weights-only)", flush=True)
        if args.patience > 0:
            callbacks.append(EarlyStopping(monitor="val/acc_macro", mode="max",
                                           patience=args.patience))

    strategy = "auto"
    if args.devices > 1:
        backend = "gloo" if platform.system() == "Windows" else "nccl"
        strategy = DDPStrategy(process_group_backend=backend,
                               find_unused_parameters=False)
        print(f"DDP backend: {backend}")

    trainer = L.Trainer(
        accelerator="gpu", devices=args.devices, strategy=strategy,
        precision="bf16-mixed", max_epochs=args.epochs,
        accumulate_grad_batches=args.accum,
        callbacks=callbacks,
        enable_checkpointing=not bool(args.benchmark_steps),
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        log_every_n_steps=50,
        limit_train_batches=args.benchmark_steps if args.benchmark_steps else 1.0,
        limit_val_batches=20 if args.benchmark_steps else 1.0,
    )
    ckpt_path = None
    if args.resume == "last":
        # prefer the active checkpoint folder; fall back to the run folder (e.g. after a crash
        # where the checkpoints were moved back into runs/<run-name>/checkpoints/ manually).
        cands = [os.path.join(ckpt_dir, "last.ckpt"), os.path.join(final_ckpt_dir, "last.ckpt")]
        ckpt_path = next((c for c in cands if os.path.exists(c)), None)
        print(f"resume: {ckpt_path or 'no last.ckpt found, starting fresh'}")
    elif args.resume:
        ckpt_path = args.resume
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # If checkpoints went to a separate folder, move them into the run folder now that training
    # finished cleanly. This final move touches the run folder (possibly 9p) and could stall, which
    # is acceptable here since the run is done; on a crash/cancel it is skipped, so move manually.
    # rank-0 only and no collectives, so it cannot desync DDP.
    if trainer.is_global_zero and ckpt_dir != final_ckpt_dir:
        os.makedirs(final_ckpt_dir, exist_ok=True)
        names = sorted(n for n in os.listdir(ckpt_dir)
                       if os.path.isfile(os.path.join(ckpt_dir, n)))
        print(f"moving {len(names)} checkpoint file(s): {ckpt_dir} -> {final_ckpt_dir}", flush=True)
        for n in names:
            shutil.move(os.path.join(ckpt_dir, n), os.path.join(final_ckpt_dir, n))
        print("checkpoint move complete", flush=True)


if __name__ == "__main__":
    main()
