"""Example usage of openseed_dump_pixels and openseed_dump_text.

This script mirrors the workflow from ``full-inmemory-test-openseed.ipynb``.
It loads OpenSeeD text and pixel embeddings and visualises a simple
segmentation based on cosine similarity.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from openseed_dump_text import openseed_text
from openseed_dump_pixels import load_model, pixel_features


def visualize_segmentation(img_path: str, feats: np.ndarray, label_embs: np.ndarray, labels: list[str], alpha: float = 0.5) -> None:
    """Overlay a segmentation obtained from the pixel and text features."""
    C, H, W = feats.shape
    flat = feats.reshape(C, -1)
    scores = label_embs @ flat
    seg = scores.reshape(len(labels), H, W).argmax(0)

    img = np.array(Image.open(img_path).convert("RGB"), dtype=float) / 255.0
    colors = plt.get_cmap("tab20", len(labels))(range(len(labels)))[:, :3]
    overlay = img * (1 - alpha) + colors[seg] * alpha

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    patches = [plt.matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo for OpenSeeD feature utilities")
    parser.add_argument("image", help="path to an input image")
    args = parser.parse_args()

    classes = ["big flat counter", "sofa", "floor", "chair", "wash basin", "picture", "other"]
    text_emb = openseed_text(classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("configs/openseed/openseed_swinl_lang_decouple.yaml", "weights/openseed_swinl_pano_sota.pt", device)
    pix_emb = pixel_features(model, args.image)

    print("pixel features:", pix_emb.shape, "text features:", text_emb.shape)
    visualize_segmentation(args.image, pix_emb, text_emb, classes, alpha=0.9)


if __name__ == "__main__":
    main()