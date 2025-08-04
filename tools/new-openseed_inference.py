# /home/jovyan/teaching_material/msc/OpenSeeD/tools/openseed_inference.py
#!/usr/bin/env python
# FINAL-V6: Removes manual normalization to match the original demo's logic, fixing performance.

import os
import sys
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Setup Paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openseed.BaseModel import BaseModel
from openseed import build_model
from detectron2.data import MetadataCatalog

# --- 1) Model Loading ---
def load_openseed_model(cfg_file, ckpt_file, device):
    cfg_path  = cfg_file if os.path.isabs(cfg_file)  else os.path.join(PROJECT_ROOT, cfg_file)
    ckpt_path = ckpt_file if os.path.isabs(ckpt_file) else os.path.join(PROJECT_ROOT, ckpt_file)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"] = ckpt_path
    cfg.setdefault("MODEL", {})["DEVICE"] = device

    model = BaseModel(cfg, build_model(cfg)) \
            .from_pretrained(ckpt_path) \
            .eval() \
            .to(device)
            
    return model

# --- Utility Hook Class ---
class PredictorInputHook:
    def __init__(self):
        self.features = None
    def __call__(self, module, input, output):
        self.features = input[1]

# --- 2) Inference Function ---
@torch.no_grad()
def segment_with_embeddings(model, image_path, labels, long_edge=512):
    # 1) Metadata and Text Embeddings
    metadata = Metadata()
    metadata.thing_classes = labels
    metadata.stuff_classes = labels
    metadata.stuff_dataset_id_to_contiguous_id = {i: i for i in range(len(labels))}
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(labels)
    encoder = model.model.sem_seg_head.predictor.lang_encoder
    encoder.get_text_embeddings(labels, is_eval=True)
    text_embs = encoder.default_text_embeddings

    # 2) Image Preprocessing - CORRECTED to match demo_semseg.py
    image_ori = Image.open(image_path).convert("RGB")
    W, H      = image_ori.size
    
    transform = transforms.Compose([transforms.Resize(long_edge, interpolation=Image.BICUBIC)])
    image_resized = transform(image_ori)
    
    # The BaseModel expects a raw, un-normalized tensor.
    device = next(model.parameters()).device
    image_tensor = torch.as_tensor(np.asarray(image_resized)).permute(2, 0, 1).to(device)
    
    inputs = [{"image": image_tensor, "height": H, "width": W}]

    # 3) Hook, Forward Pass, and Cleanup
    hook = PredictorInputHook()
    predictor = model.model.sem_seg_head.predictor
    hook_handle = predictor.register_forward_hook(hook)
    outputs = model.forward(inputs, inference_task="sem_seg")
    hook_handle.remove()

    # 4) Process Outputs
    final_outputs = outputs[-1]
    sem_seg = final_outputs["sem_seg"]
    sem_seg_up = F.interpolate(
        sem_seg.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0]
    seg_map = sem_seg_up.argmax(dim=0).cpu().numpy()

    if hook.features is None:
        raise RuntimeError("Hook failed to capture predictor input features.")
    features_256d = hook.features
    
    features_up = F.interpolate(
        features_256d, size=(H, W), mode="bilinear", align_corners=False
    )
    visual_emb = features_up[0].cpu().numpy()

    return seg_map, text_embs.cpu().numpy(), visual_emb