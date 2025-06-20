#!/usr/bin/env python
"""Utility to obtain ℓ2-normalised 512‑D text embeddings using CLIP.

This module exposes :func:`openseed_text` which mirrors the API of the
original OpenSeeD repository but only relies on the ``clip`` package.
``cfg`` and ``weight`` arguments are accepted for compatibility but are not
used.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import torch
import clip


CFG = "/home/jovyan/teaching_material/msc/OpenSeeD/configs/openseed/openseed_swinl_lang_decouple.yaml"
WGT = "/home/jovyan/teaching_material/msc/OpenSeeD/weights/openseed_swinl_pano_sota.pt"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL, _ = clip.load("ViT-B/32", device=_DEVICE)
_MODEL.eval()


def openseed_text(labels: Sequence[str], cfg: str = CFG, weight: str = WGT) -> np.ndarray:
    """Return ℓ2-normalised CLIP embeddings for *labels*.

    Parameters
    ----------
    labels:
        Sequence of strings to encode.
    cfg, weight:
        Ignored but kept for API compatibility.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(labels), 512)`` with float32 values on the CPU.
    """

    labels = [s.strip() for s in labels if s.strip()]
    if not labels:
        return np.empty((0, 512), dtype=np.float32)

    tokens = clip.tokenize(labels).to(_DEVICE)
    with torch.no_grad():
        z = _MODEL.encode_text(tokens)
        z = torch.nn.functional.normalize(z, dim=-1)

    return z.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump CLIP text embeddings")
    parser.add_argument("labels", nargs="+", help="labels to encode")
    parser.add_argument("--out", required=True, help="output .npy file")
    parser.add_argument("--cfg", default=CFG)
    parser.add_argument("--weight", default=WGT)
    args = parser.parse_args()

    embeddings = openseed_text(args.labels, args.cfg, args.weight)
    np.save(args.out, embeddings)

