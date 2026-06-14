"""Tier 2 DIAGNOSTIC -- the pose probe gave R2<0 for ALL encoders incl frozen LeWM (which the paper says
encodes pose) => the pose LABELS are almost certainly wrong (get_state picked the wrong obs field). This
dumps the PushT obs/info structure and linear-probes the FROZEN LeWM latent against EACH candidate low-dim
field, to find the real pose field (R2 >> 0) and confirm the latent encodes it. No training. Cheap.
Run on Colab GPU:  python src/tier2_diag.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401
from torchvision import transforms as TT

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402

N, T, FS = 30, 24, 5
device = "cuda" if torch.cuda.is_available() else "cpu"
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model, cfg = load_lewm("/content/le-wm", device=device)

env = gym.make("swm/PushT-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)
print("=== obs structure ===")
print("obs type:", type(obs))
if isinstance(obs, dict):
    for k, v in obs.items():
        a = np.asarray(v)
        print(f"  obs[{k!r}]: shape {a.shape} dtype {a.dtype} ex {a.ravel()[:6]}")
else:
    a = np.asarray(obs); print(f"  obs: shape {a.shape} dtype {a.dtype} ex {a.ravel()[:8]}")
print("info:", {k: np.asarray(v).shape for k, v in info.items()} if isinstance(info, dict) else info)


def cand_fields(obs, info):
    out = {}
    src = obs if isinstance(obs, dict) else {"obs": obs}
    for k, v in src.items():
        a = np.asarray(v).ravel()
        if a.size <= 32 and np.issubdtype(a.dtype, np.number):
            out[k] = a.size
    if isinstance(info, dict):
        for k, v in info.items():
            try:
                a = np.asarray(v).ravel()
                if a.size <= 32 and np.issubdtype(a.dtype, np.number):
                    out["info:" + k] = a.size
            except Exception:
                pass
    return out


def get_field(obs, info, key):
    if key.startswith("info:"):
        return np.asarray(info[key[5:]]).ravel().astype("float32")
    src = obs if isinstance(obs, dict) else {"obs": obs}
    return np.asarray(src[key]).ravel().astype("float32")


cands = list(cand_fields(obs, info).keys())
print("candidate low-dim fields:", cand_fields(obs, info))

frames = []; fields = {k: [] for k in cands}; gen = np.random.default_rng(0)
for r in range(N):
    obs, info = env.reset(seed=int(gen.integers(1_000_000_000)))
    frames.append(env.render())
    for k in cands:
        fields[k].append(get_field(obs, info, k))
    for _ in range(T):
        for _ in range(FS):
            obs, _, _, _, info = env.step(env.action_space.sample().astype("float32"))
        frames.append(env.render())
        for k in cands:
            fields[k].append(get_field(obs, info, k))
frames = np.stack(frames)


@torch.no_grad()
def enc(fr):
    Z = []
    for i in range(0, len(fr), 16):
        pix = torch.stack([prep(f) for f in fr[i:i + 16]]).unsqueeze(1).to(device)
        Z.append(model.encode({"pixels": pix})["emb"][:, 0])
    return torch.cat(Z).cpu()


Z = enc(frames).numpy().astype("float64"); n = len(Z); ntr = int(0.8 * n)


def ridge_r2(Y):                                               # closed-form ridge probe, best over alpha sweep
    Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    Ztr, Ytr, Zev, Yev = Z[:ntr], Ys[:ntr], Z[ntr:], Ys[ntr:]
    zm, ym = Ztr.mean(0), Ytr.mean(0); Zc = Ztr - zm
    best = -1e9
    for a in (1.0, 10.0, 100.0, 1000.0):
        W = np.linalg.solve(Zc.T @ Zc + a * np.eye(Zc.shape[1]), Zc.T @ (Ytr - ym))
        pred = (Zev - zm) @ W + ym
        r2 = float((1 - ((pred - Yev) ** 2).sum(0) / (((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-9)).mean())
        best = max(best, r2)
    return best


print("\n=== frozen-LeWM RIDGE-probe R2 per candidate field (closed-form, alpha-swept; real pose should be >0) ===")
for k in cands:
    Y = np.stack(fields[k]).astype("float64")
    if Y.shape[0] != n or Y.std() < 1e-6:
        print(f"  {k:18}: skip (len/const)"); continue
    print(f"  {k:18} (dim {Y.shape[1]:2}): R2 {ridge_r2(Y):+.3f}   ex {Y[0][:6]}")
print("\n-> if frozen LeWM now probes state/block_pose at R2>0, the Adam probe was overfitting:")
print("   fix tier2_pose_probe.py to RIDGE + the right field + re-run. If still <0, the latent doesn't")
print("   linearly encode pose and Tier 2's premise fails -> consolidate instead.")
