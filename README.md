# California Small Animals Classifier

Training image classifier(s) for the LILA [California Small Animals](https://lila.science/datasets/california-small-animals/) camera-trap dataset (CA Dept. of Fish & Wildlife; downward-facing, short-focus Reconyx cameras over drift fences).

## Dataset at a glance

- 2,278,071 images across **701 cameras** (`location` == folder, exactly 1:1).
- COCO Camera Traps format; metadata: `california_small_animals_with_sequences.json`.
- ~58% empty (`blank` + `misfire`); long-tailed taxonomy (257 source categories, full class/order/family/genus/species on each).
- Mostly 2048x1440 JPEGs.

See `analyze_metadata.py` / `analyze_taxonomy.py` (outputs in the output folder) for the full breakdown, and `build_label_map.py` for the source→target category mapping.

## Plan (first pass)

- **Single flat multi-class classifier** over a medium-granularity label set (**29 classes** = 27 animal + `blank` + `setup_pickup`); see `label_map.py`. Provenance (camera split + category assignments) is recorded in `training_info.20260608.json`.
- Split **by camera**, 85/15 train/val, **class-aware via ILP** (every class lands ~15% in val and appears on both sides); locked in `camera_split.csv`. See `make_split.py`.
- `blank` downsampled to **1 frame/sequence, then capped 300/camera (~115k)**. Multi-annotation images (9,372) dropped. ~1.06M images total.
- Stored training copies: whole frame resized to **~512px short side** (JPEG q90) under `F:\data\california-small-animals-training` as `train/<class>/<camera>__<id>.jpg`.
- Backbone: **timm `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`** @ 448px, trained with PyTorch Lightning on 2x RTX 4090 (DDP). Checkpoint every epoch.

### Training environment — WSL2 (not native Windows)

Native Windows PyTorch is missing FlashAttention/mem-efficient SDPA (falls back to the slow math kernel), Triton (so `torch.compile` won't run), and NCCL (gloo-only multi-GPU). Measured eva02_large@448 at only ~20 img/s/GPU on Windows. **Training therefore runs in WSL2** (Ubuntu, conda env `california-small-animals-wsl`), which has all three. Data is read from `/mnt/f`; logs/checkpoints go to `/mnt/c/temp/california-small-animals-output`. Launch via `wsl_train.sh`.

Fast config (accuracy-safe): **autocast bf16 (fp32 master) + grad-checkpointing + `torch.compile`** → ~37 img/s/GPU (vs 22 uncompiled). Notes: `torch.compile` + DDP needs `torch._dynamo.config.optimize_ddp = False` (DDPOptimizer chokes on EVA's rope/SDPA subgraph); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids compile-time fragmentation OOMs; per-GPU batch 24. Measured ~64 img/s on 2 GPUs (both ~100% util, ~12 GB each), **~4.2 h/epoch** (18,618 steps).

**Run training** (from a WSL shell — the launcher activates the env, cds into the correct folder, and sets the alloc config, and runs `train.py`):

```
bash /mnt/c/temp/california-small-animals-output/wsl_train.sh --devices 2 --batch-size 24 --workers 12 --epochs 20 --patience 5 --run-name eva02-20260101
```

`--patience 5` enables early stopping on `val/acc_macro` (stops if macro accuracy hasn't improved for 5 epochs); omit it for a fixed `--epochs` run. Each run writes everything to its own folder `runs/<run-name>/` (see "Output folder structure" below); `--run-name` defaults to a timestamp, and the launcher refuses to reuse an existing run folder — pick a new name, or pass `--resume last` to continue an interrupted one. Track progress via `runs/<run-name>/metrics.csv` (or `nvidia-smi`); the best epoch's checkpoint (by `val/acc_macro`) is the one to keep.

### Output folder structure

The base output folder `C:\temp\california-small-animals-output` (`/mnt/c/temp/...` in WSL) holds only cross-run files; everything produced by a single training run lives under `runs/<run-name>/`.

Base folder — shared across all runs (six files):
- `split.parquet` — per-image train/val assignment + resized-image paths; read by `train.py` on every run.
- `manifest.parquet` — per-image source-of-truth manifest (kept images, labels, camera, sequence); input to `make_split.py`.
- `camera_split.parquet` — the locked camera→split (train/val) assignment.
- `camera_split.csv` — the same assignment, human-readable.
- `camera_class_counts.parquet` — per-camera per-class image counts (informational).
- `wsl_train.sh` — the training launcher.

Per-run folder `runs/<run-name>/`:
- `checkpoints/` — per-epoch checkpoints (`<run-name>-NN.ckpt`) plus `last.ckpt`.
- `hparams.yaml` — the run's hyperparameters.
- `metrics.csv` — train/val metrics (loss, micro/macro accuracy, learning rate).
- `train_<run-name>.log` — full stdout/stderr log of the run.

### Preprocessing and augmentation

- **448×448 input by squashing the whole frame** — no scale-crop, no center-crop. With image-level labels and a small animal anywhere in frame, aggressive `RandomResizedCrop` would frequently crop the animal out (label noise), and val center-crop would clip edge animals. The downward-facing/baited-box geometry means low scale variation, so the whole frame at a fixed scale is fine, and train/val share the same field of view.
- **Info-banner handling.** Reconyx top/bottom banners carry timestamp + temperature (a shortcut the model could exploit, that won't transfer to other cameras) and are the only consistent orientation cue. We **crop the banner at training** (measured for this dataset) and add **synthetic-banner augmentation** (random dark bars of varying height/content at top/bottom) so the model learns to ignore arbitrary banners on *other* cameras at inference. Day/night info still comes for free from IR-grayscale vs daytime color. The inference script makes the banner crop configurable (independent top/bottom; default off — robust thanks to the synthetic-banner aug).
- **Geometric aug** (banner cropped ⇒ no canonical up/down): horizontal **and vertical** flips, 90°/mild rotation, mild affine translate/scale with reflection padding. Position jitter specifically fights camera-background memorization (cameras are static and we split by camera).
- **Photometric aug**: brightness/contrast/saturation/hue, mild blur/noise.
- Class imbalance handled at train time (balanced sampling / loss weighting), not by deleting animal data.

## TODO (near-term, this pass)

- **Inference script**: bulk inference script to emit the [MegaDetector output format](https://lila.science/megadetector-output-format), using a full-image "detection" box per image (this is a classification-only model), roughly matching the semantics of [run_md_and_speciesnet.py](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_md_and_speciesnet.py) (images only, no video).
- **Eval script**: Verify that we can replicate the validation accuracy numbers using the bulk inference script
- **Inference-time banner-crop A/B (quick):** evaluate val accuracy with the banner crop on vs off, to pick the inference default and confirm the synthetic-banner augmentation actually makes the model crop-agnostic.


## Next steps / ideas to revisit

- **Architecture A/B (if we want to push accuracy past eva02_large@448):** candidates to prioritize, roughly in order —
  1. `convnextv2_large` / `convnextv2_huge` @ 384–512 (strong fine-grained CNN, fast).
  2. `eva02_large_patch14_448` is our baseline; also try `eva02_large` at higher test-time resolution (448→512) via timm's dynamic img-size.
  3. DINOv2/DINOv3 ViT-L/g backbones with a linear/MLP head (excellent features; newer than eva02).
  4. `beit_large_patch16_512` / `swinv2_large` as additional convex points.

  Compare on the same camera-split val set; accuracy is the priority, not speed.
- **Hierarchical cascade (alternative to the flat model):** e.g. blank/non-blank → coarse class (mammal/reptile/amphibian/bird/insect/other) → finer visual groups. Could be more robust on the long tail and lets us calibrate a high-recall blank filter independently. Wrap all stages behind one inference script.
- **Banner handling as a *training-time* experiment:** compare (a) no crop, (b) crop only, (c) crop + synthetic-banner aug — on val accuracy and cross-camera robustness. Revisit whether to crop at train at all. (The inference-time A/B in the TODO list is the quick first cut of this.)
- **Rectangular-input fine-tuning** (e.g. 448×630 to match the ~1.42:1 frame aspect via pos-embed interpolation) to avoid the squash distortion entirely.
- Revisit `unknown` (736) — currently excluded; could become an abstain/OOD target.
- Consider keeping the vertebrate label on multi-annotation images to recover ~9k images.
- Test-time augmentation and per-class threshold calibration for deployment.
- Detector-guided cropping (crop to an animal box, then classify) would dissolve the whole-frame small-animal problem, but MegaDetector is unreliable on these downward-facing cameras, so it's not an option unless a suitable detector emerges.

## Dataset bugs found (reported to maintainer)

- `zebra-tailed lizard` (id 230): `order` = `"phrynosomatidae"` should be `"squamata"`.
- `western diamondback rattlesnake` (id 235): `family` = `"viperdae"` should be `"viperidae"`.
- `aspidocelis species` (id 239) is a misspelled duplicate of `aspidoscelis species` (id 158).
- `ensifera species` (id 44): 2 annotations labeled Sword-billed Hummingbird (Andean; cannot occur in CA) — likely an auto-ID artifact.
- **Corrupt image file (0 bytes):** `2002875_cemap-small-animals_exclude-identify/images/deployment/2035549/a7056e2a-66da-4610-b0b8-3623a92bf677.jpg` (image id `a7056e2a-...`, location `2002875_siteh2-hp`, labeled `rodent`). Metadata lists it as 2048x1440 but the file is empty. Dropped from training.
