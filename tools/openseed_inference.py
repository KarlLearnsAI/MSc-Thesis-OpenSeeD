# /home/jovyan/teaching_material/msc/OpenSeeD/tools/openseed_inference.py
#!/usr/bin/env python
# Extended: also extract a SPATIAL visual embedding from the model’s backbone.

import os
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Setup Paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openseed.BaseModel import BaseModel
from openseed import build_model
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata

# --- 1) Model Loading (unchanged) ---
def load_openseed_model(cfg_file, ckpt_file, device):
    cfg_path  = cfg_file if os.path.isabs(cfg_file)  else os.path.join(PROJECT_ROOT, cfg_file)
    ckpt_path = ckpt_file if os.path.isabs(ckpt_file) else os.path.join(PROJECT_ROOT, ckpt_file)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"]              = ckpt_path
    cfg.setdefault("MODEL", {})["DEVICE"] = device

    model = BaseModel(cfg, build_model(cfg)) \
                .from_pretrained(ckpt_path) \
                .eval() \
                .to(device)
    return model

# --- Define a hook class to capture features ---
class FeatureHook:
    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        # The 'predictor' module receives 'mask_features' as its second input argument.
        self.features = input[1]

    def clear(self):
        self.features = None

# --- 2) Inference + Embeddings + Visual Embedding ---
@torch.no_grad()
def extract_segmentation_and_features(model, image_path, labels, long_edge=512):
    # 1) fresh metadata
    metadata = Metadata()
    metadata.thing_classes = labels
    metadata.stuff_classes = labels
    metadata.stuff_dataset_id_to_contiguous_id = {i: i for i in range(len(labels))}
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(labels)

    # 2) text embeddings
    encoder = model.model.sem_seg_head.predictor.lang_encoder
    encoder.get_text_embeddings(labels, is_eval=True)
    text_embs = encoder.default_text_embeddings

    # 3) Load & preprocess image manually
    image_ori = Image.open(image_path).convert("RGB")
    W, H      = image_ori.size
    
    transform = transforms.Compose([
        transforms.Resize(long_edge, interpolation=Image.BICUBIC),
    ])
    image = transform(image_ori)
    image = torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1).cuda()
    inputs = [{"image": image, "height": H, "width": W}]
    
    # 4) Set up and register the hook on the predictor
    feature_hook = FeatureHook()
    predictor = model.model.sem_seg_head.predictor
    hook_handle = predictor.register_forward_hook(feature_hook)

    # 5) Run the full forward pass
    outputs = model.forward(inputs, inference_task="sem_seg")

    # 6) Remove the hook
    hook_handle.remove()

    # 7) Get the segmentation map from the final output
    sem = outputs[-1]["sem_seg"]
    sem_up = F.interpolate(
        sem.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0]
    seg_map = sem_up.argmax(dim=0).cpu().numpy()

   # 8) Get the visual embedding from the hook
    if feature_hook.features is None:
        raise RuntimeError("Hook did not capture any features.")
    
    # This is the 256-dim feature map from the model
    features_256d = feature_hook.features
    
    # --- CORRECTED PROJECTION: Use the model's trained projection layer ---
    # The class_embed layer projects from the 256-dim hidden space to the 512-dim text space.
    # Its shape is (hidden_dim, text_dim), so (256, 512).
    projection_matrix = model.model.sem_seg_head.predictor.class_embed
    
    # Reshape for matrix multiplication, apply projection, then reshape back
    # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
    b, c, h, w = features_256d.shape
    features_256d_reshaped = features_256d.permute(0, 2, 3, 1).reshape(b * h * w, c)
    
    # Apply the learned projection: (B*H*W, 256) @ (256, 512) -> (B*H*W, 512)
    features_512d_reshaped = features_256d_reshaped @ projection_matrix
    
    # Reshape back to image format: (B*H*W, 512) -> (B, H, W, 512) -> (B, 512, H, W)
    visual_emb_tensor = features_512d_reshaped.reshape(b, h, w, 512).permute(0, 3, 1, 2)
    
    visual_emb = visual_emb_tensor[0].cpu().numpy() # Remove batch dim
    
    return seg_map, text_embs.cpu().numpy(), visual_emb


# --- 3) Visualization (Improved and More Robust) ---
def visualize_segmentation(img_or_path, seg_map, labels, alpha=0.7):
    """
    Visualizes the segmentation. Can accept either a file path or a NumPy array.
    """
    if isinstance(img_or_path, str):
        # If it's a path, load it
        img = np.array(Image.open(img_or_path).convert("RGB"), dtype=float) / 255.0
    else:
        # Assume it's already a NumPy array
        img = img_or_path.astype(float) / 255.0
    
    K = len(labels)
    colors = plt.get_cmap("tab20", K)(range(K))[:, :3]
    mask = colors[seg_map]
    overlay = img * (1 - alpha) + mask * alpha

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    patches = [Patch(color=colors[i], label=labels[i]) for i in range(K)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()

# --- 3) Visualization (unchanged) ---
# def visualize_segmentation(img_path, seg_map, labels, alpha=0.7):
#     # ─── Add these lines right below the signature ───
#     K = len(labels)
#     # colors = plt.get_cmap("viridis", K)(range(K))[:, :3]
#     colors = plt.get_cmap("tab20")(np.arange(K))[:, :3]
#     mask   = colors[seg_map]
#     # ────────────────────────────────────────────────

#     img    = np.array(Image.open(img_path).convert("RGB"), dtype=float)/255.0
#     overlay = img*(1-alpha) + mask*alpha
#     plt.figure(figsize=(10,10))
#     plt.imshow(overlay); plt.axis("off")
#     patches = [Patch(color=colors[i], label=labels[i]) for i in range(K)]
#     plt.legend(handles=patches, bbox_to_anchor=(1.05,1),
#                loc="upper left", borderaxespad=0.)
#     plt.tight_layout(); plt.show()  # ← last line of the function


# --- Example CLI usage ---
if __name__ == "__main__":
    CFG     = 'configs/openseed/openseed_swinl_lang_decouple.yaml'
    WGT     = 'weights/openseed_swinl_pano_sota.pt'
    IMG     = os.path.join(PROJECT_ROOT, 'images/animals.png')
    CLASSES = ["zebra","antelope","giraffe","ostrich",
               "sky","water","grass","sand","tree"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_openseed_model(CFG, WGT, device)

    seg_map, text_embs, visual_emb = extract_segmentation_and_features(
        model, IMG, CLASSES
    )

    print("Segmentation map shape:", seg_map.shape)
    print("Text embeddings shape:", text_embs.shape)
    print("Visual embedding shape:", visual_emb.shape)

    visualize_segmentation(IMG, seg_map, CLASSES)