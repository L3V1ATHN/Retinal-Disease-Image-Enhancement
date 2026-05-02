"""
Rigorous Retinal Fundus Enhancement + ML Pipeline
No timm dependency. Uses torchvision ResNet18.

Goal:
Compare whether image enhancement improves retinal disease classification.

Experiments:
1. raw
2. clahe
3. proposed

Protocol:
- Train only on Training set
- Tune threshold/checkpoint using Evaluation set
- Report final metrics on Test set only
- Fixed random seed
- Save best checkpoint per enhancement method
"""

from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms, models

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


# =========================================================
# 1. Config
# =========================================================

class CFG:
    DATA_ROOT = Path(".")

    TRAIN_DIR = DATA_ROOT / "Training"
    EVAL_DIR  = DATA_ROOT / "Evaluation"
    TEST_DIR  = DATA_ROOT / "Test"

    TRAIN_CSV = DATA_ROOT / "Training" / "labels_train.csv"
    EVAL_CSV  = DATA_ROOT / "Evaluation" / "labels_eval.csv"
    TEST_CSV  = DATA_ROOT / "Test" / "labels_test.csv"

    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 6
    SEED = 42
    PATIENCE = 4

    # Start with binary. Later switch to "multilabel".
    TASK = "binary"

    # Run all three for the actual report.
    METHODS = ["raw", "clahe", "proposed"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR = Path("outputs")


# =========================================================
# 2. Reproducibility
# =========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================================================
# 3. Enhancement methods
# =========================================================

def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]


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
    """
    img = remove_black_border(img)
    img = apply_mask(img)

    r, g, b = cv2.split(img)

    # Illumination estimation
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=30, sigmaY=30)

    # Division-based illumination correction
    norm = g.astype(np.float32) / (bg.astype(np.float32) + 1e-6)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(norm)

    # Mild structure sharpening
    blur = cv2.GaussianBlur(enhanced_g, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(enhanced_g, 1.4, blur, -0.4, 0)

    # Blend with original green channel to avoid over-enhancement
    final_g = cv2.addWeighted(g, 0.35, sharp, 0.65, 0)

    out = cv2.merge([r, final_g, b])
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return apply_mask(out)


def enhance(img, method):
    if method == "raw":
        return img
    if method == "clahe":
        return clahe_filter(img)
    if method == "proposed":
        return proposed_filter(img)
    raise ValueError(f"Unknown method: {method}")


# =========================================================
# 4. Dataset
# =========================================================

class RetinalDataset(Dataset):
    def __init__(self, img_dir, csv_path, task="binary", enhancement="raw", augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.task = task
        self.enhancement = enhancement
        self.augment = augment

        if task == "binary":
            self.label_cols = ["Disease_Risk"]
        elif task == "multilabel":
            self.label_cols = [c for c in self.df.columns if c not in ["ID", "Disease_Risk"]]
        else:
            raise ValueError("TASK must be 'binary' or 'multilabel'")

        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def find_image_path(self, image_id):
        image_id = str(image_id)
        possible_names = [
            image_id,
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            f"{image_id}.tif",
            f"{image_id}.tiff",
        ]
        for name in possible_names:
            p = self.img_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for ID={image_id} in {self.img_dir}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.find_image_path(row["ID"])

        img = cv2.imread(str(path))
        if img is None:
            print(f"WARNING: Skipping unreadable/corrupt image: {path}")
            return None

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = enhance(img, self.enhancement)
            img = Image.fromarray(img)
            img = self.tf(img)
        except Exception as e:
            print(f"WARNING: Error processing {path}: {e}")
            return None

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels


def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# =========================================================
# 5. Model
# =========================================================

def build_model(num_outputs):
    # New torchvision API compatibility fallback
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    # Full fine-tuning: do NOT freeze layers.
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model


# =========================================================
# 6. Class imbalance handling
# =========================================================

def compute_pos_weight(csv_path, label_cols):
    df = pd.read_csv(csv_path)
    y = df[label_cols].values.astype(np.float32)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    pos_weight = negatives / (positives + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# =========================================================
# 7. Train / predict / metrics
# =========================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        if batch is None:
            continue
        x, y = batch
        x = x.to(CFG.DEVICE)
        y = y.to(CFG.DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs_all = []
    labels_all = []

    for batch in tqdm(loader, desc="predict", leave=False):
        if batch is None:
            continue
        x, y = batch
        x = x.to(CFG.DEVICE)
        logits = model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.numpy())

    return np.vstack(probs_all), np.vstack(labels_all)


def tune_threshold(probs, labels):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def metrics_from_probs(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    out = {
        "threshold": threshold,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

    if labels.shape[1] == 1:
        out["accuracy"] = accuracy_score(labels.ravel(), preds.ravel())
        try:
            out["roc_auc"] = roc_auc_score(labels.ravel(), probs.ravel())
        except ValueError:
            out["roc_auc"] = np.nan
        tn, fp, fn, tp = confusion_matrix(labels.ravel(), preds.ravel()).ravel()
        out["sensitivity_recall"] = tp / (tp + fn + 1e-8)
        out["specificity"] = tn / (tn + fp + 1e-8)
    else:
        try:
            out["roc_auc_macro"] = roc_auc_score(labels, probs, average="macro")
            out["roc_auc_weighted"] = roc_auc_score(labels, probs, average="weighted")
        except ValueError:
            out["roc_auc_macro"] = np.nan
            out["roc_auc_weighted"] = np.nan

    return out


# =========================================================
# 8. Image quality metrics
# =========================================================

def image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def image_quality_metrics(img_dir, csv_path, method, max_images=300):
    df = pd.read_csv(csv_path).head(max_images)
    entropy_vals = []
    contrast_vals = []
    sharpness_vals = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"quality-{method}", leave=False):
        image_id = str(row["ID"])
        path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            p = Path(img_dir) / f"{image_id}{ext}"
            if p.exists():
                path = p
                break
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = enhance(img, method)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        entropy_vals.append(image_entropy(gray))
        contrast_vals.append(float(gray.std()))
        sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return {
        "method": method,
        "entropy": np.mean(entropy_vals),
        "contrast_std": np.mean(contrast_vals),
        "laplacian_sharpness": np.mean(sharpness_vals),
    }


# =========================================================
# 9. Run one rigorous experiment
# =========================================================

def run_experiment(method):
    print("\n" + "=" * 70)
    print(f"Experiment: {method.upper()} | Task: {CFG.TASK}")
    print("=" * 70)

    train_ds = RetinalDataset(CFG.TRAIN_DIR, CFG.TRAIN_CSV, CFG.TASK, method, augment=True)
    eval_ds = RetinalDataset(CFG.EVAL_DIR, CFG.EVAL_CSV, CFG.TASK, method, augment=False)
    test_ds = RetinalDataset(CFG.TEST_DIR, CFG.TEST_CSV, CFG.TASK, method, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=safe_collate,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=safe_collate,
    )

    model = build_model(len(train_ds.label_cols)).to(CFG.DEVICE)

    pos_weight = compute_pos_weight(CFG.TRAIN_CSV, train_ds.label_cols).to(CFG.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_eval_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    ckpt_path = CFG.OUT_DIR / f"best_{CFG.TASK}_{method}.pt"

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        eval_probs, eval_labels = predict(model, eval_loader)
        threshold, eval_f1 = tune_threshold(eval_probs, eval_labels)
        eval_metrics = metrics_from_probs(eval_probs, eval_labels, threshold)
        scheduler.step(eval_metrics["f1_macro"])

        print(
            f"Epoch {epoch:02d}/{CFG.NUM_EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"eval_f1={eval_metrics['f1_macro']:.4f} | "
            f"eval_auc={eval_metrics.get('roc_auc', eval_metrics.get('roc_auc_macro', np.nan)):.4f} | "
            f"thr={threshold:.2f}"
        )

        if eval_metrics["f1_macro"] > best_eval_f1:
            best_eval_f1 = eval_metrics["f1_macro"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "threshold": threshold,
                "epoch": epoch,
                "eval_metrics": eval_metrics,
                "label_cols": train_ds.label_cols,
                "method": method,
                "task": CFG.TASK,
            }, ckpt_path)
        else:
            bad_epochs += 1

        if bad_epochs >= CFG.PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Load best checkpoint and evaluate on test set exactly once
    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    best_threshold = ckpt["threshold"]

    test_probs, test_labels = predict(model, test_loader)
    test_metrics = metrics_from_probs(test_probs, test_labels, best_threshold)

    print("\nBest evaluation epoch:", ckpt["epoch"])
    print("Final TEST metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    row = {
        "method": method,
        "best_epoch": ckpt["epoch"],
        "eval_best_f1": best_eval_f1,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return row


# =========================================================
# 10. Main
# =========================================================

if __name__ == "__main__":
    seed_everything(CFG.SEED)
    CFG.OUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {CFG.DEVICE}")
    print(f"Task: {CFG.TASK}")
    print(f"Image size: {CFG.IMAGE_SIZE}")

    # Image-processing metrics for report
    quality_rows = []
    for method in CFG.METHODS:
        quality_rows.append(image_quality_metrics(CFG.TRAIN_DIR, CFG.TRAIN_CSV, method))
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(CFG.OUT_DIR / f"quality_metrics_{CFG.TASK}.csv", index=False)
    print("\nImage quality metrics:")
    print(quality_df)

    # ML comparison
    results = []
    for method in CFG.METHODS:
        results.append(run_experiment(method))

    results_df = pd.DataFrame(results)
    results_df.to_csv(CFG.OUT_DIR / f"ml_results_{CFG.TASK}.csv", index=False)

    print("\nFinal comparison:")
    print(results_df)
    print(f"\nSaved outputs to: {CFG.OUT_DIR.resolve()}")
