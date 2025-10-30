# Run: python train.py --model unet --loss dice
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from pathlib import Path
import csv
import argparse

from lib.model.EncDecModel import EncDec
from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.dataset.PhCDataset import PhC

def batch_pixel_stats(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).to(targets.dtype)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct, total
    
# -------------------- NEW: args + factories --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["encdec", "unet", "unet2", "dilated"], default="encdec",
                    help="Which model to train")
parser.add_argument("--loss", choices=["bce", "dice", "focal", "bce_tv"], default="bce",
                    help="Which loss to use")
args = parser.parse_args()

MODEL_MAP = {
    "encdec": EncDec,
    "unet": UNet,
    "dilated": DilatedNet,
}
LOSS_MAP = {
    "bce": BCELoss,
    "dice": DiceLoss,
    "focal": FocalLoss,
    "bce_tv": BCELoss_TotalVariation,
}
# ---------------------------------------------------------------

# Dataset
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)),
                                     transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])

batch_size = 6
trainset = PhC(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                          num_workers=3)
testset = PhC(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=3)
# IMPORTANT NOTE: There is no validation set provided here, but don't forget to
# have one for the project

print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(testset)} test images")

# -------------------- NEW: device + chosen model/loss --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MODEL_MAP[args.model]().to(device)
#summary(model, (3, 256, 256))
learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

loss_fn = LOSS_MAP[args.loss]()  # choose loss
# ------------------------------------------------------------------------

epochs = 20

# -------------------- NEW: CSV logging setup --------------------
run_tag = f"{args.model}_{args.loss}_sz{size}_bs{batch_size}"
out_dir = Path(os.getenv("OUT_DIR", f"runs/{run_tag}"))
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "losses.csv"
with open(log_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "test_loss", "train_acc", "test_acc"])
# ----------------------------------------------------------------

# Training loop
X_test, Y_test = next(iter(test_loader))
model.train()  # train mode
for epoch in range(epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{epochs}')

    avg_loss = 0
    train_correct_pix, train_total_pix = 0, 0

    for X_batch, y_true in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        opt.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()

        avg_loss += loss / len(train_loader)
        c, t = batch_pixel_stats(y_pred, y_true, thresh=0.5)
        train_correct_pix += c
        train_total_pix   += t

    train_loss = float(avg_loss.detach().cpu())
    train_acc  = (train_correct_pix / max(train_total_pix, 1)) if train_total_pix else 0.0
    print(f' - loss: {train_loss:.4f} | train acc: {train_acc:.4f}')

    # -------- NEW: quick test loss + write CSV --------
    model.eval()
    test_loss, tb = 0.0, 0
    test_correct_pix, test_total_pix = 0, 0
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            logits = model(Xb)
            test_loss += loss_fn(logits, Yb).item()
            c, t = batch_pixel_stats(logits, Yb, thresh=0.5)
            test_correct_pix += c
            test_total_pix   += t
            tb += 1
    test_loss = test_loss / max(tb, 1)
    test_acc  = (test_correct_pix / max(test_total_pix, 1)) if test_total_pix else 0.0
    print(f'   test loss: {test_loss:.4f} | test acc: {test_acc:.4f}')
    model.train()

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch + 1, train_loss, test_loss, train_acc, test_acc])

    # ---------------------------------------------------

# Save the model
torch.save(model.state_dict(), out_dir / 'model.pth')
print(f"Training has finished! Saved to {out_dir}")
