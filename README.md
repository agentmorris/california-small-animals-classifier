# California Small Animals Classifier

Training image classifier(s) for the LILA [California Small Animals](https://lila.science/datasets/california-small-animals/) camera trap dataset.  This dataset is provided by California Fish and Wildlife); it contains images from downward-facing, short-focus Reconyx cameras over drift fences.

This README is not intended (yet) as an external introduction to the project, it's intended as internal book-keeping, so, excuse all the local filenames that have no meaning other than the computer where I'm typing this.

## Dataset at a glance

- 2,278,071 images across 701 cameras (each leaf-node folder is a camera deployment/location)
- Metadata in COCO Camera Traps format (`california_small_animals_with_sequences.json`)
- ~58% empty (images labeled as `blank` or `misfire`)
- Long-tailed taxonomy (257 source categories, full class/order/family/genus/species on each)
- Mostly 2048x1440 JPEGs
- Raw dataset is in `e:\data\california-small-animals`

See `analyze_metadata.py` / `analyze_taxonomy.py` (in the output folder) for the full breakdown, and `build_label_map.py` for the source→target category mapping.

## Plan

- *Single, flat multi-class classifier* over a medium-granularity label set (29 classes = 27 animal + `blank` + `setup_pickup`); see `label_map.py`. Provenance (camera split + category assignments) is recorded in `training_info.20260608.json`.  Here we use "single, flat" to contrast this approach with a hierarchical approach that we may pursue in the future, where a first model classifies, e.g., blank/non-blank, or blank/mammal/reptile/bird/amphibian, etc., and subsequent, taxa-specific models classify species.
- Split *by camera*, 85/15 train/val, *class-aware via ILP* (every class lands ~15% in val and appears on both sides).  Locked in `camera_split.csv`. See `make_split.py`.
- `blank` downsampled to *1 frame/sequence, then capped 300/camera (~115k)*. Multi-annotation images (9,372) dropped. ~1.06M images total.
- Stored training copies: whole frame resized to *~512px short side* (JPEG q90) under `F:\data\california-small-animals-training` as `train/<class>/<camera>__<id>.jpg`.
- Backbone: *timm `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`* @ 448px, trained with PyTorch Lightning on 2x RTX 4090 (DDP). Checkpoint every epoch.

### Training environment

Native Windows PyTorch is missing FlashAttention/mem-efficient SDPA (falls back to the slow math kernel), Triton (so `torch.compile` won't run), and NCCL (gloo-only multi-GPU). Measured eva02_large@448 at only ~20 img/s/GPU on Windows. Training therefore runs in WSL (Ubuntu, conda env `california-small-animals-wsl`), which has all three. Data is read from `/mnt/f`; logs/checkpoints go to `/mnt/c/temp/california-small-animals-output`. Launch via `wsl_train.sh`.

Fast config (accuracy-safe): autocast bf16 (fp32 master) + grad-checkpointing + `torch.compile` → \~37 img/s/GPU (vs 22 uncompiled). 

Performance notes: `torch.compile` + DDP needs `torch._dynamo.config.optimize_ddp = False` (DDPOptimizer chokes on EVA's rope/SDPA subgraph); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids compile-time fragmentation OOMs; per-GPU batch 24. Measured \~64 img/s on 2 GPUs (both \~100% util, \~12 GB each), \~4.2 h/epoch (18,618 steps).

### Starting Training

From a WSL shell; the launcher activates the env, cds into the correct folder, and sets the alloc config, and runs `train.py`)...

```
bash /mnt/c/temp/california-small-animals-output/wsl_train.sh --devices 2 --batch-size 24 --workers 12 --epochs 20 --patience 5 --run-name eva02-20260628
```

`--patience 5` enables early stopping on `val/acc_macro` (stops if macro accuracy hasn't improved for 5 epochs); omit it for a fixed `--epochs` run. Each run writes everything to its own folder `runs/<run-name>/` (see "Output folder structure" below); `--run-name` defaults to a timestamp, and the launcher refuses to reuse an existing run folder — pick a new name, or pass `--resume last` to continue an interrupted one. Track progress via `runs/<run-name>/metrics.csv` (or `nvidia-smi`); the best epoch's checkpoint (by `val/acc_macro`) is the one to keep.

### Monitoring training

#### Debug output

```bash
export RUN_NAME="eva02-20260628"
watch -n 10 tail -n 20 /mnt/c/temp/california-small-animals-output/runs/${RUN_NAME}/train_${RUN_NAME}.log
```

#### Accuracy metrics

Training losses are updated every batch, validation metrics are written every epoch.

```bash
export RUN_NAME="eva02-20260628"
watch -n 10 tail -n 20 /mnt/c/temp/california-small-animals-output/runs/${RUN_NAME}/metrics.csv
```

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

Archive folder (`archive`) — one-off scripts, exploratory logs, and throwaway debug/verification outputs. Anything clearly not reusable across runs should go *straight here* rather than the base folder, so the base folder stays limited to the cross-run files listed above. Obvious one-offs — diagnostic/launcher scripts, ad-hoc test or invariance outputs, logs from exploratory work — belong in `archive`; when it's genuinely unclear whether an output will be reused, it's fine to leave it in the base folder. Nothing in `archive` is deleted automatically; the maintainer prunes it manually.  This folder is interchangeably referred to as the "archive folder" or the "scratch folder".

### Preprocessing and augmentation

- **448×448 input by squashing the whole frame**.  No scale-crop, no center-crop. With image-level labels and a small animal anywhere in frame, aggressive `RandomResizedCrop` would frequently crop the animal out, and val center-crop would clip edge animals. The downward-facing/baited-box geometry means low scale variation, so the whole frame at a fixed scale is fine, and train/val share the same field of view.
- **Info-banner handling.** Reconyx top/bottom banners carry timestamp + temperature (a shortcut the model could exploit, that won't transfer to other cameras) and are the only consistent orientation cue. We *crop the banner at training* (measured for this dataset) and add *synthetic-banner augmentation* (random dark bars of varying height/content at top/bottom) so the model learns to ignore arbitrary banners on *other* cameras at inference. Day/night info still comes for free from IR-grayscale vs daytime color. The inference script makes the banner crop configurable (independent top/bottom; default off — robust thanks to the synthetic-banner aug).
- **Geometric aug** (banner cropped ⇒ no canonical up/down): horizontal and vertical flips, 90°/mild rotation, mild affine translate/scale with reflection padding. Position jitter specifically fights camera-background memorization (cameras are static and we split by camera).
- **Photometric aug**: brightness/contrast/saturation/hue, mild blur/noise.
- Class imbalance handled at train time (balanced sampling / loss weighting), not by deleting animal data.

## Inference-ready checkpoints

`strip_checkpoint.py` converts a Lightning training checkpoint into a compact, self-describing `*.stripped.ckpt` (optimizer/scheduler/callback state removed; ~3.6 GB → ~1.2 GB). It is a plain dict saved with `torch.save` (load with `weights_only=False`); everything except `state_dict` is configuration metadata, i.e. plain numbers/strings, not code or a graph. `run_inference.py` reads these fields to rebuild the model and its preprocessing. Stored fields:

- `format` — schema tag (`csa-classifier-inference-v1`).
- `model_name` — the timm model id used to recreate the architecture.
- `num_classes` — number of output classes.
- `classes` — the ordered list of class names; the list index is the model's output label id.
- `img_size` — the square input size the model expects (e.g. 448).
- `norm_mean`, `norm_std` — per-channel normalization (from the timm model's data config).
- `banner_crop` — `{top, bottom}` fractions of image height that were cropped off the top/bottom during training (a record of the training-time preprocessing; the inference script's own banner handling is configured separately).
- `preprocessing` — a human-readable description of the preprocessing pipeline.
- `weights_dtype` — `float32` or `float16` (the dtype the weights are stored in).
- `source_checkpoint`, `epoch`, `global_step` — provenance: which training checkpoint this was derived from.
- `state_dict` — the timm model weights (the only non-metadata payload).

## Label review / data cleanup

After the first training run we ran a label-cleanup pass focused on *blank ↔ non-blank* ground-truth errors — the most common and most consequential mislabels in this dataset — using the model's own high-confidence disagreements to prioritize what a human looks at. The review outcome is stored back with the data (`E:\data\california-small-animals\manual_review_<date>.json`) because it is a property of the dataset, not of any one run. The analysis/review scripts live in the analysis workspace `C:\temp\california-small-animals-output\archive\data-review\`; the pipeline scripts that consume the result (`copy_resize.py`, `make_gt_coco.py`, `make_val_cct.py`) are in this repo.

1. **Find candidates.** Run the chosen model over the full train+val set (`run_inference.py`) and, with `analyze_blank_confusion.py`, tabulate every image whose top-1 prediction disagrees with the folder label on blank-vs-animal, bucketed by confidence. It writes an HTML report (`blank_confusion_analysis.html`) with per-class / per-confidence-bucket counts where each non-zero count links to a gallery of the actual images (no overlaid annotations) for a quick eyeball.
2. **Stage full-size images for review.** `copy_review_images.py` copies the full-size originals of every blank↔non-blank mismatch (confidence ≥ 0.5) from the raw data folder into `C:\temp\california-small-animals-image-review`, laid out as `[label]/[prediction]/[bucket]/[location]_[datetime]_[framenum]_[guid].jpg` so they sort by camera then time. It writes `review_manifest.csv` (maps every review path back to the source image + metadata), and `make_timelapse_csv.py` derives a Timelapse-ready `review_manifest_timelapse.csv` (adds `File`/`RelativePath` columns).
3. **Adjudicate in Timelapse.** Images are reviewed in [Timelapse](https://timelapse.ucalgary.ca/); the reviewer marks an image `incorrect` when the *ground-truth label* is wrong (e.g. labeled `blank` but an animal is clearly present, or vice-versa). An empty outcome means "no decision" (not necessarily confirmed-correct); a `correct` tag is available but has no training impact.
4. **Record outcomes.** `process_review.py` verifies the Timelapse export used only expected tags, maps the `incorrect` rows back to original filenames via the manifest, and writes `manual_review_<date>.json` = `{ "<original/relpath>": "incorrect", ... }` (the dict form leaves room for additional outcome tags later).
5. **Re-train on the cleaned set, same split.** The exclusion list is centralized in `label_map.py` (`EXCLUDE_FILES` + `load_excluded_guids()`), and both `copy_resize.py` (regenerating the training tree) and `train.py` (building the train/val image lists in `load_frames()`) drop the flagged images — so the folder and the trainer can't drift, and `split.parquet` stays the immutable locked split (the camera assignment is reused unchanged). `assess_split_impact.py` first confirms no class loses a split or drops to ≤1 location. After regenerating the tree, `make_gt_coco.py` and `make_val_cct.py` rebuild the master train+val GT (`california-small-animals-training.json`) and the val-only GT (`val/val_cct.json`). Both the copy step and the trainer accept `--no-exclude` to fall back to the full uncleaned set if ever needed.

First pass (2026-06-28): 5,743 images excluded — almost all `blank`-labeled frames that actually contained an animal, plus a few animal-labeled frames that were actually empty. The prior training tree was preserved as `F:\data\california-small-animals-training-2026.06.00`.

## TODO

### P0

- **Data cleanup (blank ↔ non-blank).** First pass DONE (2026-06-28); see "Label review / data cleanup". 5,743 mislabeled images excluded; re-training on the cleaned set. Future: extend the same review workflow to non-blank confusions and other high-confidence disagreements.
- **Inference-time banner-crop A/B (quick):** evaluate val accuracy with the banner crop on vs off, to pick the inference default and confirm the synthetic-banner augmentation actually makes the model crop-agnostic.
- **Test on Ohio Small Animals data, consider adding to training**
- **Test on CCER Small Animals data, consider adding to training**
- **Test on CIC Shrew Monitoring data, consider adding to training**

### P1

- **Add checkpointing to inference script**
- **Layer freezing:**: training appears to be overfitting quickly, consider keeping eva02_large@448 but freezing many layers
- **Reduce penalties for partial mistakes**: consider adjusting the loss function so that for, e.g., a specific rodent species, predicting "other rodent" is penalized less than predicting "bird"
- **New CDFW data:** Talk to CDFW about pulling in additional data
- **Architecture A/B (if we want to push accuracy past eva02_large@448):** candidates to prioritize, roughly in order —
  1. `convnextv2_large` / `convnextv2_huge` @ 384–512 (strong fine-grained CNN, fast).
  2. `eva02_large_patch14_448` is our baseline; also try `eva02_large` at higher test-time resolution (448→512) via timm's dynamic img-size.
  3. DINOv2/DINOv3 ViT-L/g backbones with a linear/MLP head (excellent features; newer than eva02).
  4. `beit_large_patch16_512` / `swinv2_large` as additional convex points.

  Compare on the same camera-split val set; accuracy is the priority, not speed.
- **Hierarchical cascade (alternative to the flat model):** e.g. blank/non-blank → coarse class (mammal/reptile/amphibian/bird/insect/other) → finer visual groups. Could be more robust on the long tail and lets us calibrate a high-recall blank filter independently. Wrap all stages behind one inference script.
- **Banner handling as a *training-time* experiment:** compare (a) no crop, (b) crop only, (c) crop + synthetic-banner aug — on val accuracy and cross-camera robustness. Revisit whether to crop at train at all. (The inference-time A/B in the TODO list is the quick first cut of this.)
- **Rectangular-input fine-tuning** (e.g. 448×630 to match the ~1.42:1 frame aspect via pos-embed interpolation) to avoid the squash distortion entirely.

### P2

- Revisit `unknown` (736) — currently excluded; could become an abstain/OOD target.
- Consider keeping the vertebrate label on multi-annotation images to recover ~9k images.

## Dataset bugs found

- `zebra-tailed lizard` (id 230): `order` = `"phrynosomatidae"` should be `"squamata"`.
- `western diamondback rattlesnake` (id 235): `family` = `"viperdae"` should be `"viperidae"`.
- `aspidocelis species` (id 239) is a misspelled duplicate of `aspidoscelis species` (id 158).
