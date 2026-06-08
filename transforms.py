"""Preprocessing / augmentation for the California Small Animals classifier.

Pipeline (train):
  full frame -> crop real banner (kills timestamp/temp shortcut)
             -> squash to SxS
             -> dihedral aug (H/V flip + k*90 rotation; no fill needed)
             -> small-angle rotation + mild affine translate/scale
             -> with prob p: add a SYNTHETIC banner (random dark bar top/bottom)
                so the model learns to ignore arbitrary banners on other cameras
             -> photometric (color jitter, occasional blur)
             -> tensor + normalize
Val:
  full frame -> (optionally) crop banner -> squash SxS -> tensor + normalize
The val banner-crop is a flag so we can A/B "crop vs no-crop" at inference time.

Camera is downward-facing => no canonical up/down once the banner is gone, so the
full dihedral group is label-preserving. See README.
"""
import random

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_banner(img, top_frac, bot_frac):
    """Remove the top/bottom info-banner bands (fractions of height)."""
    w, h = img.size
    t = int(round(h * top_frac))
    b = int(round(h * bot_frac))
    if t + b >= h:
        return img
    return img.crop((0, t, w, h - b))


def _add_synthetic_banner(img, max_frac=0.06):
    """Overlay a content-free dark bar at top and/or bottom to mimic a foreign
    camera's info banner (so the model learns to ignore such bars)."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for edge in ("top", "bottom"):
        if random.random() < 0.35:           # sometimes only one / neither edge
            continue
        bh = int(round(h * random.uniform(0.02, max_frac)))
        if bh < 2:
            continue
        shade = random.randint(0, 40)         # near-black bar
        if edge == "top":
            box = (0, 0, w, bh)
            ty = (0, bh)
        else:
            box = (0, h - bh, w, h)
            ty = (h - bh, h)
        draw.rectangle(box, fill=(shade, shade, shade))
        # sparse bright "text"-like marks
        for _ in range(random.randint(0, 18)):
            x = random.randint(0, max(0, w - 6))
            y = random.randint(ty[0], max(ty[0], ty[1] - 2))
            ln = random.randint(2, 8)
            v = random.randint(180, 255)
            draw.line([(x, y), (x + ln, y)], fill=(v, v, v), width=1)
    return img


class TrainTransform:
    def __init__(self, img_size=448, banner_top=0.03, banner_bot=0.035,
                 p_synth_banner=0.5, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.s = img_size
        self.bt = banner_top
        self.bb = banner_bot
        self.p_banner = p_synth_banner
        self.mean = mean
        self.std = std
        self.jitter = ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.03)

    def __call__(self, img):
        img = crop_banner(img, self.bt, self.bb)
        img = img.resize((self.s, self.s), Image.BILINEAR)        # squash

        # --- dihedral (fill-free) ---
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k)                              # exact, no fill

        # --- mild continuous jitter ---
        angle = random.uniform(-7, 7)
        tx = random.uniform(-0.06, 0.06) * self.s
        ty = random.uniform(-0.06, 0.06) * self.s
        scale = random.uniform(0.9, 1.1)
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=scale,
                        shear=0.0, interpolation=TF.InterpolationMode.BILINEAR)

        # --- synthetic banner (cross-camera banner robustness) ---
        if random.random() < self.p_banner:
            img = _add_synthetic_banner(img)

        # --- photometric ---
        img = self.jitter(img)
        if random.random() < 0.15:
            img = TF.gaussian_blur(img, kernel_size=3)

        t = TF.to_tensor(img)
        if random.random() < 0.15:
            t = t + torch.randn_like(t) * random.uniform(0.0, 0.03)
            t = t.clamp(0, 1)
        return TF.normalize(t, self.mean, self.std)


class ValTransform:
    def __init__(self, img_size=448, banner_top=0.03, banner_bot=0.035,
                 crop_banner_flag=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.s = img_size
        self.bt = banner_top
        self.bb = banner_bot
        self.crop = crop_banner_flag
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if self.crop:
            img = crop_banner(img, self.bt, self.bb)
        img = img.resize((self.s, self.s), Image.BILINEAR)
        t = TF.to_tensor(img)
        return TF.normalize(t, self.mean, self.std)
