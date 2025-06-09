# OpenSeeD/tools/openseed_dump_pixels.py
#!/usr/bin/env python
"""
Extract ℓ2-normalised CLIP pixel embeddings from OpenSeeD.

Works with every public checkpoint (swint, swin-large, vit-l, …)
because it:
  • Inserts PROJECT_ROOT into sys.path (no pip install needed)
  • Patches LanguageEncoder (dummy text embedding)
  • Patches OpenSeeDEncoder (adds a real .forward)
"""

# ── repo path helper ───────────────────────────────────────────────────────
import os, sys, argparse, yaml, torch, numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)              #  ← makes “import openseed” work

from openseed import build_model
from openseed.BaseModel import BaseModel
import types, torch, inspect

# ── two tiny patches ───────────────────────────────────────────────────────
def _ensure_dummy_text_embeddings(root_mod, device):
    for m in root_mod.modules():
        if m.__class__.__name__ == "LanguageEncoder":
            if hasattr(m, "embed_dim"):
                dim = m.embed_dim
            elif hasattr(m, "text_proj"):          # newer checkpoints
                dim = m.text_proj.in_features
            else:
                dim = 512
            if not hasattr(m, "default_text_embeddings"):
                with torch.no_grad():
                    z = torch.zeros(1, dim, device=device)
                    m.default_text_embeddings = torch.nn.functional.normalize(z, dim=1)
            break



def _patch_openseed_encoder(root_mod):
    """
    For every sub-module that has a .transformer_forward method
    (all OpenSeeD encoder variants), alias it to .forward.
    """
    for m in root_mod.modules():
        if hasattr(m, "transformer_forward"):
            m.forward = types.MethodType(m.transformer_forward, m)


# ── model loader ───────────────────────────────────────────────────────────
def load_model(cfg_file, ckpt, device):
    # resolve paths first
    if not os.path.isabs(cfg_file):
        cfg_file = os.path.join(PROJECT_ROOT, cfg_file)
    if not os.path.isabs(ckpt):
        ckpt     = os.path.join(PROJECT_ROOT, ckpt)

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"] = ckpt
    cfg.setdefault("MODEL", {})["DEVICE"] = device

    model = BaseModel(cfg, build_model(cfg)).from_pretrained(ckpt).eval().to(device)

    _ensure_dummy_text_embeddings(model.model, device)
    _patch_openseed_encoder(model.model)
    return model

# ── feature extractor  ─────────────────────────────────────────────────────
@torch.no_grad()
def pixel_features(model, img_path, long_edge=512):
    img = Image.open(img_path).convert("RGB")
    H, W = img.height, img.width
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC),
                 ToTensor()])(img) * 255
    device = next(model.parameters()).device
    feats = model.model.backbone(x.to(device).unsqueeze(0))

    # public API, with dummy masks
    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out

    mask_feats = torch.nn.functional.interpolate(
        mask_feats, size=(H, W), mode="bilinear", align_corners=False)[0]
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()          # (C,H,W)

# ── new helper to accept an in-memory array ────────────────────────────────
@torch.no_grad()
def pixel_features_from_array(model, img_array: np.ndarray, long_edge: int = 512) -> np.ndarray:
    """
    Exactly like `pixel_features`, but takes a H×W×3 uint8 array instead of a file path.
    Returns a (C,H,W) float32 array, L2-normalized across C.
    """
    # 1) Convert NumPy array (H,W,3, uint8) to PIL Image
    pil_img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
    H, W = pil_img.height, pil_img.width

    # 2) Resize & tensorize
    x = Compose([Resize(long_edge, interpolation=Image.BICUBIC),
                 ToTensor()])(pil_img) * 255
    device = next(model.parameters()).device

    # 3) Backbone + pixel decoder (exactly as before)
    feats = model.model.backbone(x.to(device).unsqueeze(0))
    out = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out

    # 4) Upsample back to original H×W and L2-normalize
    mask_feats = torch.nn.functional.interpolate(
        mask_feats, size=(H, W), mode="bilinear", align_corners=False)[0]
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=0)
    return mask_feats.cpu().numpy()           # (C,H,W)

# ── CLI  ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument("--cfg",    default="configs/openseed/openseed_swint_lang.yaml")
    p.add_argument("--cfg",    default="configs/openseed/openseed_swinl_lang_decouple.yaml")
    # p.add_argument("--weight", default="weights/model_state_dict_swint_51.2ap.pt")
    p.add_argument("--weight", default="weights/openseed_swinl_pano_sota.pt")
    p.add_argument("--image",  required=True)
    p.add_argument("--out",    required=True)
    args = p.parse_args()

    dev  = "cuda" if torch.cuda.is_available() else "cpu"
    mdl  = load_model(args.cfg, args.weight, dev)
    np.save(args.out, pixel_features(mdl, args.image))