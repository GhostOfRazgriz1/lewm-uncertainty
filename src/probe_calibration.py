"""OOD calibration probe: does LeWM's SIGReg-Gaussian latent flag inputs it doesn't know?
In-distribution Push-T frames (from le-wm's demo gif) vs OOD (other envs, noise, corruption). Signal =
per-sample ||emb||; AUC(in-dist vs OOD). (Carried from the perceptor exploration; M1 upgrades this to a
PREDICTIVE-error calibration using real planning-rollout transitions.)
Usage: python src/probe_calibration.py [LEWM_DIR]   (default ../le-wm)"""
import sys
import numpy as np
import torch
from PIL import Image, ImageSequence
from torchvision import transforms as TT
from load_lewm import load_lewm

LEWM_DIR = sys.argv[1] if len(sys.argv) > 1 else "../le-wm"
model, cfg = load_lewm(LEWM_DIR)
prep = TT.Compose([TT.Resize((224, 224)), TT.ToTensor(), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

gif = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(f"{LEWM_DIR}/assets/lewm.gif"))]
rng = list(range(774, 858, 2))
boxes = {"pusht (in-dist)": (60, 375, 196, 511), "robot-arm (OOD env)": (430, 375, 650, 511),
         "reacher (OOD env)": (792, 375, 935, 511)}
sources = {n: torch.stack([prep(gif[i].crop(b)) for i in rng if np.array(gif[i].crop(b)).std() > 12])
           for n, b in boxes.items()}
base = sources["pusht (in-dist)"]
sources["noise (OOD)"] = torch.stack([prep(Image.fromarray((np.random.default_rng(s).random((128, 128, 3)) * 255).astype("uint8")))
                                      for s in range(len(base))])
g = torch.Generator().manual_seed(0)
shuf = base.clone().reshape(base.shape[0], 3, -1)
for i in range(shuf.shape[0]):
    shuf[i] = shuf[i][:, torch.randperm(shuf.shape[2], generator=g)]
sources["shuffled (OOD)"] = shuf.reshape(base.shape)

with torch.no_grad():
    norm = {n: model.encode({"pixels": x.unsqueeze(1)})["emb"][:, 0].norm(dim=1).numpy() for n, x in sources.items()}


def auc(a, b):
    s = np.concatenate([a, b]); lab = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
    npos, nneg = lab.sum(), (1 - lab).sum()
    return (ranks[lab == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)


ind = norm["pusht (in-dist)"]
print(f"in-dist ||emb|| {ind.mean():.2f} (Gaussian shell ~{cfg['predictor']['input_dim'] ** 0.5:.1f})")
for n in sources:
    if n != "pusht (in-dist)":
        print(f"  vs {n:22s}: ||emb|| {norm[n].mean():6.2f} | AUC {auc(ind, norm[n]):.3f} (|AUC-.5|={abs(auc(ind, norm[n]) - .5):.2f})")
