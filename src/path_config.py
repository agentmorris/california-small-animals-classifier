"""Per-machine / per-environment path configuration.

All machine-specific absolute paths live in a small JSON file (passed to scripts via
``--path-config``), not in the code, so the same scripts run unchanged on any machine. Note this
is per *environment*, not just per machine: the original box runs training under WSL (``/mnt/...``
paths) but eval under native Windows (``C:\\...`` paths), so it keeps two config files; a native
Linux box needs only one.

Required keys (all must be present):
  METADATA_FILE - COCO Camera Traps metadata JSON
  IMAGE_ROOT    - root of the original (full-size) image tree
  OUTPUT_ROOT   - output base folder (runs/, split.parquet, manifest.parquet, ... live here)
  TRAIN_ROOT    - root of the resized train/val tree
  EXCLUDE_FILES - list of manual-review JSON files; images marked "incorrect" are dropped from
                  training. Use [] for none, but the key itself is still required.

See ``path_config.example.json`` for a template.
"""
import json
import os
from types import SimpleNamespace

REQUIRED = ("METADATA_FILE", "IMAGE_ROOT", "OUTPUT_ROOT", "TRAIN_ROOT", "EXCLUDE_FILES")


def load_path_config(path):
    """Load and validate a path-config JSON file; returns a namespace with the REQUIRED keys."""
    if not path:
        raise SystemExit("a --path-config JSON file is required")
    if not os.path.isfile(path):
        raise SystemExit(f"path-config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"path-config {path} is missing required key(s): {', '.join(missing)}")
    if not isinstance(cfg["EXCLUDE_FILES"], list):
        raise SystemExit("path-config: EXCLUDE_FILES must be a list of paths (use [] for none)")
    return SimpleNamespace(**{k: cfg[k] for k in REQUIRED})


def load_excluded_guids(exclude_files):
    """Collect image guids flagged 'incorrect' across one or more manual-review JSON files.

    Each file maps an original relative path -> outcome; the guid is the filename stem. Used by
    copy_resize.py and train.py so both apply the same exclusions.
    """
    guids = set()
    for p in exclude_files:
        if not p or not os.path.exists(p):
            print(f"  (exclusion file not found, skipping: {p})")
            continue
        with open(p, encoding="utf-8") as f:
            review = json.load(f)
        n = sum(1 for o in review.values() if o == "incorrect")
        for relpath, outcome in review.items():
            if outcome == "incorrect":
                guids.add(os.path.splitext(os.path.basename(relpath))[0])
        print(f"  loaded {n:,} 'incorrect' exclusions from {p}")
    return guids
