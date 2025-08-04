#!/usr/bin/env python
"""Dump L2-normalized text embeddings using OpenSeeD model's language encoder, matching demo_semseg.py."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import yaml
import torch
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
def openseed_text(model, labels: Sequence[str]) -> np.ndarray:
    labels = [s.strip() for s in labels if s.strip()]
    if not labels:
        return np.empty((0, 512), dtype=np.float32)

    # Use model's language encoder, matching demo_semseg.py
    embeddings = model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(labels, is_eval=True)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump text embeddings using OpenSeeD model")
    parser.add_argument("labels", nargs="+", help="labels to encode")
    parser.add_argument("--out", required=True, help="output .npy file")
    parser.add_argument("--cfg", default="configs/openseed/openseed_swinl_lang_decouple.yaml")
    parser.add_argument("--weight", default="weights/openseed_swinl_pano_sota.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = load_model(args.cfg, args.weight, device)
    embeddings = openseed_text(mdl, args.labels)
    np.save(args.out, embeddings)