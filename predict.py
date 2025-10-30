# imports the model architecture
# loads the saved weights: Use torch.load function
# loads the test set of a DatasetLoader (see train.py)
# Iterate over the test set images, generate predictions, save segmentation masks

# RUN: python predict.py --weights model.pth --outdir pred_masks --img_size 128 --batch_size 8 --num_workers 4


from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# your modules
from lib.model.EncDecModel import EncDec   # or UNet, etc.
from lib.dataset.PhCDataset import PhC     # the same dataset used in train.py

def save_mask(array, path):
    # array should be a 2D numpy array with 0s and 1s
    np.unique(array) == [0, 1]
    len(np.shape(array)) == 2
    im_arr = (array*255)
    Image.fromarray(np.uint8(im_arr)).save(path)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="model.pth",
                    help="Path to model state_dict (.pth)")
    ap.add_argument("--outdir", type=str, default="pred_masks",
                    help="Where to save predicted masks")
    ap.add_argument("--img_size", type=int, default=128,
                    help="Resize used at test time (must match training)")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="encdec",
                    choices=["encdec", "unet", "dilated"])
    return ap.parse_args()

def build_model(name: str):
    name = name.lower()
    if name == "encdec":
        return EncDec()
    elif name == "unet":
        from lib.model.UNetModel import UNet
        return UNet()
    elif name == "dilated":
        from lib.model.DilatedNetModel import DilatedNet
        return DilatedNet()
    else:
        raise ValueError(f"Unknown model {name}")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    # If you saved state_dict directly (as in your train.py):
    model.load_state_dict(ckpt)
    model.eval()

    # 2) Data
    tf = T.Compose([T.Resize((args.img_size, args.img_size)),
                    T.ToTensor()])
    test_ds = PhC(train=False, transform=tf)   # matches train.py
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # helper to get ID from filename (works with your PhC)
    def id_from_path(p):
        # e.g. ".../img_00017.jpg" -> "00017"
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        return stem.split("_")[1]

    # 3) Inference loop
    with torch.no_grad():
        idx_global = 0
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)              # [B,1,H,W] logits
            probs  = torch.sigmoid(logits)  # [B,1,H,W]
            preds  = (probs > args.threshold).float()  # binary

            # save each mask with a filename that matches its image id
            bsz = preds.shape[0]
            for i in range(bsz):
                # grab the original image path to build an ID
                img_path = test_ds.image_paths[idx_global]
                idx_global += 1
                ident = id_from_path(img_path)
                out_path = out_dir / f"pred_{ident}.png"
                mask = preds[i, 0].detach().cpu().numpy().astype(np.uint8)
                save_mask(mask, out_path)

    print(f"Saved predictions to: {out_dir}")

if __name__ == "__main__":
    main()