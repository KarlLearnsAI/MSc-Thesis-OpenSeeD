# OpenSeed/tools/openseed_dump_text.py
#!/usr/bin/env python

"""Dump L2-normalised 512-D text embeddings, matching demo/demo_semseg.py."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import torch
from openseed_dump_pixels import load_model  # Reuse from pixel script for consistency

def openseed_text(labels: Sequence[str], cfg: str, weight: str) -> np.ndarray:
    """Return L2-normalised text embeddings from an OpenSeeD checkpoint."""

    labels = [s.strip() for s in labels if s.strip()]
    if not labels:
        return np.empty((0, 512), dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(cfg, weight, device)

    # Match demo's text embedding call
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(labels, is_eval=True)
    emb = model.model.sem_seg_head.predictor.lang_encoder.default_text_embeddings
    return emb.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump text embeddings using an OpenSeeD checkpoint")
    parser.add_argument("labels", nargs="+", help="labels to encode")
    parser.add_argument("--out", required=True, help="output .npy file")
    parser.add_argument("--cfg", required=True, help="config YAML path")
    parser.add_argument("--weight", required=True, help="checkpoint path")
    args = parser.parse_args()

    embeddings = openseed_text(args.labels, args.cfg, args.weight)
    np.save(args.out, embeddings)
    