import os, torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lib.model.EncDecModel import EncDec
from lib.dataset.PhCDataset  import PhC
from lib.losses import BCELoss

os.environ["PHC_ROOT"] = "/dtu/datasets1/02516/phc_data"  # optional if your loader uses it

tf = T.Compose([T.Resize((128,128)), T.ToTensor()])
ds = PhC(train=True, transform=tf)
ds = torch.utils.data.Subset(ds, range(min(16, len(ds))))   # 16 samples only
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                pin_memory=False, persistent_workers=False)

device = torch.device("cpu")   # login node: CPU only
model = EncDec().to(device)
loss_fn = BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x, y = next(iter(dl))
x, y = x.to(device), y.to(device)
opt.zero_grad()
y_pred = model(x)
loss = loss_fn(y_pred, y)
loss.backward()
opt.step()
print("Smoke test OK. loss=", float(loss.detach().cpu()))