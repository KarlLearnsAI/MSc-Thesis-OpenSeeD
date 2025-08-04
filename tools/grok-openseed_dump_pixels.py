#!/usr/bin/env python
"""Extract L2-normalized CLIP pixel embeddings from OpenSeeD checkpoints, matching demo_semseg.py."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import yaml
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from typing import Sequence

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from openseed import build_model
from openseed.BaseModel import BaseModel

def load_model(cfg_file, ckpt, device):
    if not os.path.isabs(cfg_file):
        cfg_file = os.path.join(PROJECT_ROOT, cfg_file)
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(PROJECT_ROOT, ckpt)

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"] = ckpt
    cfg.setdefault("MODEL", {})["DEVICE"] = device

    model = BaseModel(cfg, build_model(cfg)).from_pretrained(ckpt).eval().to(device)
    return model

@torch.no_grad()
def pixel_features(model, img_path, long_edge=512):
    img = Image.open(img_path).convert("RGB")
    H, W = img.height, img.width
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC), ToTensor()])(img) * 255
    device = next(model.parameters()).device
    feats = model.model.backbone(x.to(device).unsqueeze(0))

    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out

    mask_feats = torch.nn.functional.interpolate(mask_feats, size=(H, W), mode="bilinear", align_corners=False)[0]

    proj = model.model.sem_seg_head.predictor.class_embed
    flat = mask_feats.flatten(1)
    if proj.shape[0] == mask_feats.shape[0]:
        mapped = proj.t().matmul(flat)
    elif proj.shape[1] == mask_feats.shape[0]:
        mapped = proj.matmul(flat)
    else:
        raise ValueError(f"class_embed shape {tuple(proj.shape)} incompatible with {mask_feats.shape[0]}-D mask features")
    mask_feats = mapped.view(proj.shape[1], H, W)

    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()  # (E, H, W)

@torch.no_grad()
def pixel_features_from_array(model, img_array: np.ndarray, long_edge: int = 512) -> np.ndarray:
    pil_img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
    H, W = pil_img.height, pil_img.width
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC), ToTensor()])(pil_img) * 255
    device = next(model.parameters()).device
    feats = model.model.backbone(x.to(device).unsqueeze(0))
    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out
    mask_feats = torch.nn.functional.interpolate(mask_feats, size=(H, W), mode="bilinear", align_corners=False)[0]
    proj = model.model.sem_seg_head.predictor.class_embed
    flat = mask_feats.flatten(1)
    if proj.shape[0] == mask_feats.shape[0]:
        mapped = proj.t().matmul(flat)
    elif proj.shape[1] == mask_feats.shape[0]:
        mapped = proj.matmul(flat)
    else:
        raise ValueError(f"class_embed shape {tuple(proj.shape)} incompatible with {mask_feats.shape[0]}-D mask features")
    mask_feats = mapped.view(proj.shape[1], H, W)
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()  # (E, H, W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump per-pixel OpenSeeD features projected to CLIP space")
    parser.add_argument("--cfg", default="configs/openseed/openseed_swinl_lang_decouple.yaml")
    parser.add_argument("--weight", default="weights/openseed_swinl_pano_sota.pt")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", required=True, help="output .npy file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = load_model(args.cfg, args.weight, device)
    np.save(args.out, pixel_features(mdl, args.image))
