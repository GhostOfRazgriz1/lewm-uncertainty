"""(1) JEPA-latent uncertainty-CALIBRATION objective: does it improve long-horizon latent fidelity,
calibration, and OOD robustness vs a plain ensemble? (Mechanism proof on frozen LeWM / PushT.)

Frozen LeWM encoder. Train an ensemble of M action-conditioned residual-MLP predictors f_i(z_t,a_t)->z_{t+1}
on frozen LeWM latents. Two variants, identical except the loss:
  baseline -- k-step autoregressive rollout MSE only (M2.1's plain ensemble, already decent).
  ours     -- confidence-weighted rollout MSE (down-weight high-disagreement steps, WIMLE-style)
              + lambda * Gaussian-NLL calibration on the SIGReg latent (ties ensemble variance to REALIZED
              error -- the principled calibration, NOT HAUWM's grow-with-horizon L_HCU which we showed is
              harmful in JEPA). Variance floor + clip for stability (the HCU-divergence lesson).

EVAL (the three axes):
  (a) fidelity   -- k-step rollout error of the ensemble mean vs horizon k (the localized bottleneck).
  (b) calibration-- within-horizon Spearman(disagreement, realized error) + does it improve over baseline.
  (c) OOD        -- on corrupted current frames, AUROC(clean vs corrupted) from disagreement / shell / both.

WIN if ours < baseline on long-horizon error AND ours >= baseline on calibration AND OOD AUROC stays high.
Spec: docs/calib-objective-spec.md.  Run on Colab GPU:  python src/calib_predictor.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
from torchvision import transforms as TT

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402

N_ROLLOUTS, T_STEPS, FS = 150, 20, 5
N_OOD = 20                                                        # rollouts kept WITH frames for the OOD test
M, K_MAX = 6, 8                                                   # ensemble size; rollout horizon
EPOCHS, BS, LR = 50, 256, 1e-3
LAM, BETA, VFLOOR = 0.5, 1.0, 1e-3                                # calib weight; confidence sharpness; var floor
NOISE_SIGMA = 0.4
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SHELL = cfg["predictor"]["input_dim"] ** 0.5
torch.manual_seed(0); np.random.seed(0)


def rollout(env, gen):
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]; acts = []
    for _ in range(T_STEPS):
        blk = [env.action_space.sample().astype("float32") for _ in range(FS)]
        for a in blk:
            env.step(a)
        acts.append(np.concatenate(blk)); frames.append(env.render())
    return np.stack(frames), np.stack(acts)


@torch.no_grad()
def encode_all(frames):
    out = []
    for i in range(0, len(frames), 32):
        pix = torch.stack([prep(f) for f in frames[i:i + 32]]).unsqueeze(1).to(device)
        out.append(model.encode({"pixels": pix})["emb"][:, 0])
    return torch.cat(out)                                         # [.,192]


def corrupt(frame, rng):
    f = frame.astype("float32") + rng.normal(0, NOISE_SIGMA * 255, frame.shape)
    return np.clip(f, 0, 255).astype("uint8")


class Pred(nn.Module):                                            # residual MLP: f(z,a) = z + g(z,a)
    def __init__(self, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192 + 10, 256), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(256, 256), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(256, 192))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


def ensemble_rollout(members, z0, acts):                          # z0 [B,192], acts [B,k,10] -> preds [M,B,k,192]
    outs = []
    for p in members:
        z = z0; seq = []
        for t in range(acts.shape[1]):
            z = p(z, acts[:, t]); seq.append(z)
        outs.append(torch.stack(seq, 1))
    return torch.stack(outs)


def train(ours):
    members = nn.ModuleList([Pred() for _ in range(M)]).to(device)
    opt = torch.optim.Adam(members.parameters(), lr=LR)
    for m in members:
        m.train()
    idx = [(r, t) for r in range(NTR) for t in range(T_STEPS - K_MAX + 1)]
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            z0 = Ztr[[r for r, _ in b], [t for _, t in b]]                       # [B,192]
            acts = torch.stack([Atr[r, t:t + K_MAX] for r, t in b])             # [B,k,10]
            tgt = torch.stack([Ztr[r, t + 1:t + K_MAX + 1] for r, t in b])      # [B,k,192]
            preds = ensemble_rollout(members, z0, acts)                         # [M,B,k,192]
            memb_mse = ((preds - tgt[None]) ** 2).mean(-1)                      # [M,B,k]
            if not ours:
                loss = memb_mse.mean()
            else:
                mu = preds.mean(0); s = preds.var(0).mean(-1)                   # [B,k,192], [B,k]
                se = ((mu - tgt) ** 2).mean(-1)                                 # [B,k] realized per-dim error
                sf = s.clamp(min=VFLOOR)
                nll = 0.5 * (se / sf + torch.log(sf))                           # calibration: var -> realized err
                w = 1.0 / (1.0 + BETA * s.detach())                            # confidence weight (down-weight unsure)
                loss = (w[None] * memb_mse).mean() + LAM * nll.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(members.parameters(), 5.0); opt.step()
    for m in members:
        m.eval()
    return members


@torch.no_grad()
def eval_fidelity_calib(members):
    starts = [(r, t) for r in range(NTR, N_ROLLOUTS) for t in range(T_STEPS - K_MAX + 1)]
    z0 = Z[[r for r, _ in starts], [t for _, t in starts]]
    acts = torch.stack([A[r, t:t + K_MAX] for r, t in starts])
    tgt = torch.stack([Z[r, t + 1:t + K_MAX + 1] for r, t in starts])
    preds = ensemble_rollout(members, z0, acts)                                # [M,B,k,192]
    mu = preds.mean(0); s = preds.var(0).mean(-1)                              # [B,k,192],[B,k]
    err = (mu - tgt).norm(dim=-1)                                              # [B,k] rollout error per step
    fid = err.mean(0).cpu().numpy()                                            # [k]
    se = ((mu - tgt) ** 2).mean(-1).cpu().numpy(); sv = s.cpu().numpy()        # [B,k] each
    cal = np.array([spearman(sv[:, k], se[:, k]) for k in range(K_MAX)])       # within-horizon Spearman per k
    return fid, cal


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


def auroc(score, lab):                                            # higher score should -> label 1 (corrupted)
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(len(score))
    p = lab == 1; npos, nneg = int(p.sum()), int((~p).sum())
    return float((ranks[p].sum() - npos * (npos - 1) / 2) / (npos * nneg + 1e-9)) if npos and nneg else float("nan")


@torch.no_grad()
def eval_ood(members):
    dis_c, dis_o, sh_c, sh_o = [], [], [], []
    for fr, ac in OOD:
        zc = encode_all(fr); zo = encode_all([corrupt(f, OODRNG) for f in fr])           # [T+1,192] clean/corrupt
        a = ac.to(device)                                                                 # [T,10]
        pc = ensemble_rollout(members, zc[:-1], a[:, None])[:, :, 0]                      # one-step from clean input
        po = ensemble_rollout(members, zo[:-1], a[:, None])[:, :, 0]
        dis_c += pc.var(0).mean(-1).cpu().tolist(); dis_o += po.var(0).mean(-1).cpu().tolist()
        sh_c += (zc[:-1].norm(dim=-1) - SHELL).abs().cpu().tolist()
        sh_o += (zo[:-1].norm(dim=-1) - SHELL).abs().cpu().tolist()
    dis = np.array(dis_c + dis_o); sh = np.array(sh_c + sh_o)
    lab = np.array([0] * len(dis_c) + [1] * len(dis_o))
    comb = (dis - dis.mean()) / (dis.std() + 1e-9) + (sh - sh.mean()) / (sh.std() + 1e-9)
    return auroc(dis, lab), auroc(sh, lab), auroc(comb, lab)


# ---- collect + encode ----------------------------------------------------------------------------
print("collecting + encoding LeWM/PushT rollouts ...", flush=True)
gen = np.random.default_rng(0); OODRNG = np.random.default_rng(7)
Zs, As, OOD = [], [], []
for r in range(N_ROLLOUTS):
    frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), gen)
    Zs.append(encode_all(frames)); As.append(torch.tensor(acts, device=device))
    if r >= N_ROLLOUTS - N_OOD:
        OOD.append((frames, torch.tensor(acts, device=device)))                          # keep frames for OOD test
    if r % 30 == 0:
        print(f"  rollout {r}/{N_ROLLOUTS}", flush=True)
Z = torch.stack(Zs); A = torch.stack(As)                                                  # [N,T+1,192],[N,T,10]
NTR = N_ROLLOUTS - 30
Ztr, Atr = Z[:NTR], A[:NTR]
print(f"  Z {tuple(Z.shape)}  train {NTR} / eval {N_ROLLOUTS-NTR} rollouts (shell={SHELL:.2f})\n", flush=True)

# ---- train both variants + eval ------------------------------------------------------------------
out = {}
for ours in (False, True):
    name = "ours (calib+conf)" if ours else "baseline (plain ens)"
    members = train(ours)
    fid, cal = eval_fidelity_calib(members)
    ood = eval_ood(members)
    out[name] = (fid, cal, ood)
    print(f"[{name}]", flush=True)
    print(f"  fidelity (rollout err @k=1..{K_MAX}): {np.round(fid,3).tolist()}", flush=True)
    print(f"  calibration Spearman @k:             {np.round(cal,3).tolist()}  (mean {cal.mean():+.3f})", flush=True)
    print(f"  OOD AUROC  disagreement {ood[0]:.3f} | shell {ood[1]:.3f} | combined {ood[2]:.3f}\n", flush=True)

# ---- verdict -------------------------------------------------------------------------------------
bf, bc, bo = out["baseline (plain ens)"]
of, oc, oo = out["ours (calib+conf)"]
long_gain = bf[-1] - of[-1]; long_rel = 100 * long_gain / (bf[-1] + 1e-9)                # k=K_MAX error drop
cal_gain = oc.mean() - bc.mean()
print("==== verdict (JEPA-latent calibration objective) ====")
print(f"  (a) long-horizon fidelity (k={K_MAX}): ours {of[-1]:.3f} vs baseline {bf[-1]:.3f}  "
      f"=> {long_gain:+.3f} ({long_rel:+.1f}%)")
print(f"  (b) calibration (mean within-horizon Spearman): ours {oc.mean():+.3f} vs baseline {bc.mean():+.3f}"
      f"  => {cal_gain:+.3f}")
print(f"  (c) OOD AUROC combined: ours {oo[2]:.3f} vs baseline {bo[2]:.3f}")
better_fid = long_gain > 0.02 * bf[-1]; better_cal = cal_gain > 0.02; ood_ok = oo[2] > 0.7
if better_fid and better_cal and ood_ok:
    print("  => POSITIVE: the calibration objective improves long-horizon fidelity AND calibration, OOD intact.")
elif better_fid or better_cal:
    print("  => PARTIAL: improves one of fidelity/calibration but not both cleanly (see rows).")
else:
    print("  => NULL: the objective does not beat the plain ensemble (the plain ensemble was already strong).")
