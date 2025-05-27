# OpenSeed/tools/openseed_dump_text.py
#!/usr/bin/env python
"""
Dump ℓ2-normalised 256-D text embeddings (CLIP-512 → truncate).
Accepts --cfg / --weight only for API compatibility.
"""

import argparse, numpy as np, torch, clip

# ---------- argument parser ------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--cfg",    default=None, help="(ignored)")
p.add_argument("--weight", default=None, help="(ignored)")
p.add_argument("--labels", required=True,
               help='comma-separated list, e.g. "door,chair,ground"')
p.add_argument("--out",    required=True)
args = p.parse_args()

# ---------- make 256-D text embeddings -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

labels = [s.strip() for s in args.labels.split(",") if s.strip()]
tok    = clip.tokenize(labels).to(device)

with torch.no_grad():
    z = model.encode_text(tok)                         # (K,512)
    z = torch.nn.functional.normalize(z, dim=-1)       # ℓ2
z = z[:, :256].cpu().numpy()                           # truncate → 256
np.save(args.out, z)