"""
Standalone Retinal Fundus Image Enhancement Visualizer

Usage:
    python visual.py --image path/to/image.png

Example:
    python visual.py --image Test/12.png

Outputs are saved in:
    enhancement_outputs/<image_name>/

Generated outputs:
    - raw.png
    - clahe.png
    - proposed.png
    - comparison.png
    - grayscale_histograms.png
    - edge_maps.png
    - metrics.csv

Dependencies:
    sudo apt install python3-opencv python3-numpy python3-pandas python3-matplotlib
or:
    python3 -m pip install opencv-python numpy pandas matplotlib
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Enhancement methods
# =========================================================

def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y + h, x:x + w]


def apply_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 15)
    return cv2.bitwise_and(img, img, mask=mask)


def clahe_filter(img):
    img = remove_black_border(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    return apply_mask(out)


def proposed_filter(img):
    """
    Proposed Adaptive Green-Channel Illumination-Corrected Enhancement Filter.

    Steps:
    1. Remove black border.
    2. Apply retinal mask.
    3. Extract green channel.
    4. Estimate illumination background using Gaussian blur.
    5. Correct illumination by division normalization.
    6. Apply CLAHE to corrected green channel.
    7. Apply mild unsharp masking.
    8. Blend enhanced green channel back into RGB image.
    """
    img = remove_black_border(img)
    img = apply_mask(img)

    r, g, b = cv2.split(img)

    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=30, sigmaY=30)

    norm = g.astype(np.float32) / (bg.astype(np.float32) + 1e-6)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(norm)

    blur = cv2.GaussianBlur(enhanced_g, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(enhanced_g, 1.4, blur, -0.4, 0)

    final_g = cv2.addWeighted(g, 0.35, sharp, 0.65, 0)

    out = cv2.merge([r, final_g, b])
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return apply_mask(out)


# =========================================================
# 2. Metrics
# =========================================================

def image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def contrast_std(gray):
    return float(gray.std())


def laplacian_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_strength(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag.mean())


def compute_metrics(name, img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "method": name,
        "entropy": image_entropy(gray),
        "contrast_std": contrast_std(gray),
        "laplacian_sharpness": laplacian_sharpness(gray),
        "edge_strength": edge_strength(gray),
    }


# =========================================================
# 3. Saving helpers
# =========================================================

def save_rgb(path, img):
    path = Path(path)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def make_comparison(images, out_path):
    names = list(images.keys())

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(names, 1):
        plt.subplot(1, len(names), i)
        plt.imshow(images[name])
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_histograms(images, out_path):
    plt.figure(figsize=(10, 6))

    for name, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        plt.plot(hist, label=name)

    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_edge_maps(images, out_path):
    names = list(images.keys())

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(names, 1):
        gray = cv2.cvtColor(images[name], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        plt.subplot(1, len(names), i)
        plt.imshow(edges, cmap="gray")
        plt.title(f"{name} edges")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_metric_barplots(metrics_df, out_path):
    metric_cols = ["entropy", "contrast_std", "laplacian_sharpness", "edge_strength"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, metric in zip(axes, metric_cols):
        ax.bar(metrics_df["method"], metrics_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Main processing
# =========================================================

def process_single_image(image_path, output_root="enhancement_outputs"):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    images = {
        "raw": raw,
        "clahe": clahe_filter(raw),
        "proposed": proposed_filter(raw),
    }

    out_dir = Path(output_root) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in images.items():
        save_rgb(out_dir / f"{name}.png", img)

    make_comparison(images, out_dir / "comparison.png")
    make_histograms(images, out_dir / "grayscale_histograms.png")
    make_edge_maps(images, out_dir / "edge_maps.png")

    metrics = [compute_metrics(name, img) for name, img in images.items()]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    make_metric_barplots(metrics_df, out_dir / "metric_barplots.png")

    print("\nSaved outputs to:", out_dir.resolve())
    print("\nMetrics:")
    print(metrics_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retinal image enhancement visualizer")
    parser.add_argument("--image", required=True, help="Path to input retinal fundus image")
    parser.add_argument("--output", default="enhancement_outputs", help="Output folder")
    args = parser.parse_args()

    process_single_image(args.image, args.output)

