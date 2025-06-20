#!/usr/bin/env python
"""Extract ℓ2-normalised CLIP pixel embeddings from OpenSeeD checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import yaml
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)  # allow ``import openseed`` without install

from openseed import build_model
from openseed.BaseModel import BaseModel
import types


def _ensure_dummy_text_embeddings(root_mod: torch.nn.Module, device: str) -> None:
    """Populate ``default_text_embeddings`` with zeros if missing."""
    for m in root_mod.modules():
        if m.__class__.__name__ == "LanguageEncoder":
            if hasattr(m, "embed_dim"):
                dim = m.embed_dim
            elif hasattr(m, "text_proj"):
                dim = m.text_proj.in_features
            else:
                dim = 512
            if not hasattr(m, "default_text_embeddings"):
                with torch.no_grad():
                    z = torch.zeros(1, dim, device=device)
                    m.default_text_embeddings = torch.nn.functional.normalize(z, dim=1)
            break


def _patch_openseed_encoder(root_mod: torch.nn.Module) -> None:
    """Alias ``transformer_forward`` to ``forward`` for encoder modules."""
    for m in root_mod.modules():
        if hasattr(m, "transformer_forward"):
            m.forward = types.MethodType(m.transformer_forward, m)


def load_model(cfg_file: str, ckpt: str, device: str) -> BaseModel:
    """Load an OpenSeeD model given a config file and checkpoint."""
    if not os.path.isabs(cfg_file):
        cfg_file = os.path.join(PROJECT_ROOT, cfg_file)
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(PROJECT_ROOT, ckpt)

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"] = ckpt
    cfg.setdefault("MODEL", {})["DEVICE"] = device

    model = BaseModel(cfg, build_model(cfg)).from_pretrained(ckpt).eval().to(device)

    _ensure_dummy_text_embeddings(model.model, device)
    _patch_openseed_encoder(model.model)
    return model


@torch.no_grad()
def pixel_features(model: BaseModel, img_path: str, long_edge: int = 512) -> np.ndarray:
    """Return ℓ2-normalised per-pixel features for ``img_path``."""
    img = Image.open(img_path).convert("RGB")
    H, W = img.height, img.width
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC), ToTensor()])(img) * 255

    device = next(model.parameters()).device
    feats = model.model.backbone(x.to(device).unsqueeze(0))

    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out

    mask_feats = torch.nn.functional.interpolate(
        mask_feats, size=(H, W), mode="bilinear", align_corners=False
    )[0]
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()  # (C, H, W)


@torch.no_grad()
def pixel_features_from_array(
    model: BaseModel, img_array: np.ndarray, long_edge: int = 512
) -> np.ndarray:
    """Same as :func:`pixel_features` but taking a ``numpy`` array image."""

    pil_img = Image.fromarray(img_array.astype("uint8"), mode="RGB")
    H, W = pil_img.height, pil_img.width
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC), ToTensor()])(pil_img) * 255

    device = next(model.parameters()).device
    feats = model.model.backbone(x.to(device).unsqueeze(0))
    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out

    mask_feats = torch.nn.functional.interpolate(
        mask_feats, size=(H, W), mode="bilinear", align_corners=False
    )[0]
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump per-pixel OpenSeeD features")
    parser.add_argument("--cfg", default="configs/openseed/openseed_swinl_lang_decouple.yaml")
    parser.add_argument("--weight", default="weights/openseed_swinl_pano_sota.pt")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", required=True, help="output .npy file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = load_model(args.cfg, args.weight, device)
    np.save(args.out, pixel_features(mdl, args.image))

