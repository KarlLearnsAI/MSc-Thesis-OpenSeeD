# OpenSeed/tools/openseed_dump_text.py
#!/usr/bin/env python
"""
Dump l2-normalised 512-D text embeddings.
Accepts --cfg / --weight only for API compatibility.
"""
import numpy as np
import torch
import clip

CFG    = "/home/jovyan/teaching_material/msc/OpenSeeD/configs/openseed/openseed_swinl_lang_decouple.yaml"
WGT    = "/home/jovyan/teaching_material/msc/OpenSeeD/weights/openseed_swinl_pano_sota.pt"
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _ = clip.load("ViT-B/32", device=_device)
_model.eval()

def openseed_text(labels, cfg: str = CFG, weight: str = WGT) -> np.ndarray:
    """
    labels: sequence of strings
    cfg, weight: kept for compatibility but not used
    returns: (K, 512) l2-normalised embeddings as a NumPy array
    """
    # clean up labels
    labels = [s.strip() for s in labels if s.strip()]

    # tokenize + move to GPU/CPU
    tokens = clip.tokenize(labels).to(_device)

    # forward + normalize
    with torch.no_grad():
        z = _model.encode_text(tokens)              # (K,512)
        z = torch.nn.functional.normalize(z, dim=-1)

    return z.cpu().numpy()