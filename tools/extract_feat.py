#!/usr/bin/env python

# ─── make “import openseed” work without pip install ─────────────────────────
import os, sys
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# ─── all the real imports ───────────────────────────────────────────────────
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from openseed import build_model
from openseed.BaseModel import BaseModel

# ─── defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG     = os.path.join(
    PROJECT_ROOT, "configs", "openseed", "openseed_swint_lang.yaml"
)
DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "weights", "model_state_dict_swint_51.2ap.pt"
)

# ─── model loader ──────────────────────────────────────────────────────────
def load_model(opt: dict, weight: str, device: str):
    opt["WEIGHT"] = weight
    opt.setdefault("MODEL", {})["DEVICE"] = device
    m = BaseModel(opt, build_model(opt))
    return m.from_pretrained(weight).eval().to(device)

# ─── feature extractor ─────────────────────────────────────────────────────
@torch.no_grad()
def get_pixels(model, img_path: str):
    # 1) load & resize
    img = Image.open(img_path).convert("RGB")
    img = transforms.Resize(512, interpolation=Image.BICUBIC)(img)
    h, w = img.size[1], img.size[0]

    # 2) to CHW tensor on model's device
    tens = torch.as_tensor(np.asarray(img)).permute(2,0,1)
    device = next(model.parameters()).device
    tens = tens.to(device)
    batch = [{"image": tens, "height": h, "width": w}]

    # 3) try dump_patches first, else semantic_inference
    feats_dict = None
    dp = getattr(model.model, "dump_patches", None)
    if callable(dp):
        try:
            feats_dict = dp(batch)
        except TypeError:
            feats_dict = None

    if feats_dict is None or not isinstance(feats_dict, dict):
        # fallback to semantic_inference
        si = getattr(model.model, "semantic_inference", None)
        if callable(si):
            feats_dict = si(batch)
        else:
            raise RuntimeError(
                "Neither dump_patches nor semantic_inference "
                "returned a dict. Available on model.model:\n  "
                + ", ".join([m for m in dir(model.model) if not m.startswith("_")])
            )

    # 4) grab the pixel tokens
    if "pixel_decoder_out" not in feats_dict:
        raise KeyError(
            f"'pixel_decoder_out' not in returned dict; keys = {list(feats_dict.keys())}"
        )
    feats = feats_dict["pixel_decoder_out"]

    # 5) upsample & normalize
    feats = torch.nn.functional.interpolate(
        feats, size=(h, w), mode="bilinear", align_corners=False
    )
    feats = torch.nn.functional.normalize(feats, dim=1)

    return feats.cpu().numpy()  # (1, C, H, W)

# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Dump per-pixel OpenSeeD CLIP embeddings to a .npy"
    )
    p.add_argument("--config-file", default=DEFAULT_CFG,
                   help="which YAML under configs/openseed to use")
    p.add_argument("--image", required=True,
                   help="input RGB image path")
    p.add_argument("--out",   required=True,
                   help="where to write the .npy")
    p.add_argument("--weight", default=DEFAULT_WEIGHTS,
                   help="override the checkpoint (.pt) to load")
    args = p.parse_args()

    # load config
    with open(args.config_file) as f:
        opt = yaml.load(f, Loader=yaml.Loader)

    # pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load & run
    model = load_model(opt, args.weight, device)
    feats = get_pixels(model, args.image)
    np.save(args.out, feats)

if __name__ == "__main__":
    main()
