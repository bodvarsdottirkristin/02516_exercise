# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)

# Run: python measure.py --pred_dir pred_masks --img_size 128
import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

from lib.dataset.PhCDataset import PhC  # uses the same test split

def binarize(arr):
    # supports 0/255 or 0/1
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)

def confusion_stats(pred, true):
    # pred/true: 2D arrays with {0,1}
    pred = pred.ravel().astype(np.uint8)
    true = true.ravel().astype(np.uint8)
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    return tp, tn, fp, fn

def metrics_from_counts(tp, tn, fp, fn, eps=1e-6):
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    sens = tp / (tp + fn + eps)            # recall / TPR
    spec = tn / (tn + fp + eps)
    dice = (2*tp) / (2*tp + fp + fn + eps)
    iou  = tp / (tp + fp + fn + eps)
    return acc, sens, spec, dice, iou

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, default="pred_masks",
                    help="Folder with predicted masks (pred_XXXXX.png)")
    ap.add_argument("--img_size", type=int, default=128,
                    help="Resize used at test time (must match predict)")
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()

def main():
    args = get_args()
    pred_dir = Path(args.pred_dir)
    assert pred_dir.exists(), f"Missing folder {pred_dir}"

    # build test dataset to get GT masks & IDs
    tf = T.Compose([T.Resize((args.img_size, args.img_size)),
                    T.ToTensor()])
    test_ds = PhC(train=False, transform=tf)

    # Mapping: id -> ground-truth array
    gt_map = {}
    for img_path, lab_path in zip(test_ds.image_paths, test_ds.label_paths):
        ident = os.path.splitext(os.path.basename(img_path))[0].split("_")[1]  # "00017"
        gt = Image.open(lab_path)
        gt_map[ident] = binarize(gt)

    # Iterate predicted files and accumulate metrics
    counts = np.zeros(4, dtype=np.float64)  # tp, tn, fp, fn summed over images
    n_ok = 0
    missing = []

    for ident, gt in gt_map.items():
        pred_path = pred_dir / f"pred_{ident}.png"
        if not pred_path.exists():
            missing.append(str(pred_path))
            continue
        pred = Image.open(pred_path)
        pred = binarize(pred)
        tp, tn, fp, fn = confusion_stats(pred, gt)
        counts += np.array([tp, tn, fp, fn], dtype=np.float64)
        n_ok += 1

    tp, tn, fp, fn = counts
    acc, sens, spec, dice, iou = metrics_from_counts(tp, tn, fp, fn)

    print(f"Evaluated {n_ok} images; missing {len(missing)} predictions.")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Sensitivity:  {sens:.4f}")
    print(f"Specificity:  {spec:.4f}")
    print(f"Dice:         {dice:.4f}")
    print(f"IoU:          {iou:.4f}")

    if missing:
        print("\nMissing examples (first 10):")
        for p in missing[:10]:
            print("  ", p)

if __name__ == "__main__":
    main()


