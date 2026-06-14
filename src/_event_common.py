"""Shared pieces for the Event-JEPA follow-ups (C2 pixel / C3 fair planner / C4 transfer).

Self-contained: a configurable numpy event-world (known sparse causal events: pickup/drop/switch),
the sparse-additive event-bottleneck model, and discovery metrics. No external deps beyond numpy/torch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

E_NAMES = ["none", "pickup", "drop", "switch"]
SDIM, ADIM, NEV = 6, 2, 4
STEP, R_PICK = 0.06, 0.12


class EventEnv:
    """state = [agent_x, agent_y, obj_x, obj_y, carrying, switch].  Zones configurable (for transfer)."""

    def __init__(self, drop_c=(0.85, 0.15), sw_c=(0.15, 0.85), zone_r=0.15, obj_box=(0.1, 0.9)):
        self.drop_c = np.array(drop_c, "float32"); self.sw_c = np.array(sw_c, "float32")
        self.zone_r = zone_r; self.obj_box = obj_box

    def reset(self, g):
        return np.array([*g.uniform(0.1, 0.9, 2), *g.uniform(*self.obj_box, 2), 0.0, 0.0], "float32")

    def step(self, s, a):
        s = s.copy(); s[:2] = np.clip(s[:2] + a, 0, 1); ev = 0
        if s[4] < 0.5 and np.linalg.norm(s[:2] - s[2:4]) < R_PICK:
            s[4] = 1.0; ev = 1
        elif s[4] > 0.5 and np.linalg.norm(s[:2] - self.drop_c) < self.zone_r:
            s[4] = 0.0; ev = 2
        if s[4] > 0.5:
            s[2:4] = s[:2]
        if ev == 0 and s[5] < 0.5 and np.linalg.norm(s[:2] - self.sw_c) < self.zone_r:
            s[5] = 1.0; ev = 3
        return s, ev

    def policy(self, s, g):
        if g.random() < 0.5:
            target = s[2:4] if s[4] < 0.5 else (self.drop_c if g.random() < 0.5 else self.sw_c)
            d = target - s[:2]; return (d / (np.linalg.norm(d) + 1e-9) * STEP).astype("float32")
        return (g.uniform(-1, 1, ADIM) * STEP).astype("float32")

    def collect(self, g, n_ep, T):
        Z, A, Zn, EV, rolls = [], [], [], [], []
        for _ in range(n_ep):
            s = self.reset(g); ep = [s]
            for _ in range(T):
                a = self.policy(s, g); s2, ev = self.step(s, a)
                Z.append(s); A.append(a); Zn.append(s2); EV.append(ev); ep.append(s2); s = s2
            rolls.append(np.array(ep))
        return (np.array(Z, "float32"), np.array(A, "float32"), np.array(Zn, "float32"), np.array(EV)), rolls


def mlp(din, dout, hid, layers=2):
    seq = [nn.Linear(din, hid), nn.GELU()]
    for _ in range(layers - 1):
        seq += [nn.Linear(hid, hid), nn.GELU()]
    return nn.Sequential(*seq, nn.Linear(hid, dout))


class EventBN(nn.Module):
    """continuous base B(z,a) + sparse additive event correction E(z,a,e); discrete code, prior+posterior."""

    def __init__(self, sdim, adim, hid, K):
        super().__init__(); self.K = K
        self.base = mlp(sdim + adim, sdim, hid)
        self.post = mlp(sdim + adim + sdim, K, hid)
        self.prior = mlp(sdim + adim, K, hid)
        self.eff = mlp(sdim + adim + K, sdim, hid)

    def forward(self, z, a, zn, tau=1.0):
        pl = self.post(torch.cat([z, a, zn], -1)); e = F.gumbel_softmax(pl, tau=tau, hard=True)
        dze = self.eff(torch.cat([z, a, e], -1))
        return self.base(torch.cat([z, a], -1)) + dze, pl, self.prior(torch.cat([z, a], -1)), dze

    def code(self, z, a):
        return self.prior(torch.cat([z, a], -1)).argmax(-1)

    def step(self, z, a):
        c = self.code(z, a)
        return self.base(torch.cat([z, a], -1)) + self.eff(torch.cat([z, a, F.one_hot(c, self.K).float()], -1)), c

    def post_code(self, z, a, zn):
        return self.post(torch.cat([z, a, zn], -1)).argmax(-1)


def train_bn(Z, A, Zn, sdim, adim, hid, K, epochs, device, rng, lr=2e-3):
    m = EventBN(sdim, adim, hid, K).to(device); opt = torch.optim.Adam(m.parameters(), lr)
    Zt, At, Znt = (torch.tensor(x, device=device, dtype=torch.float32) for x in (Z, A, Zn)); dZ = Znt - Zt
    idx = np.arange(len(Z))
    for ep in range(epochs):
        rng.shuffle(idx); tau = max(0.5, 1.0 - ep / epochs)
        for i in range(0, len(Z), 512):
            b = idx[i:i + 512]
            dz, pl, prl, dze = m(Zt[b], At[b], Znt[b], tau)
            pbar = F.softmax(pl, -1).mean(0)
            loss = ((dz - dZ[b]) ** 2).mean() + F.cross_entropy(prl, pl.detach().argmax(-1)) \
                + 0.01 * dze.abs().mean() + 0.05 * (pbar * torch.log(pbar + 1e-9)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def nmi(a, b):
    a, b = a.astype(int), b.astype(int)
    j = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(j, (a, b), 1.0); j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    mi = np.sum(j[j > 0] * np.log(j[j > 0] / (pa[:, None] * pb[None, :])[j > 0] + 1e-12))
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    Ha, Hb = ent(pa), ent(pb)
    return 0.0 if Ha == 0 or Hb == 0 else float(mi / ((Ha + Hb) / 2))


def event_metrics(true, codes, K):
    """per-event recall to a dedicated code + enrichment lift (NMI is none-dominated)."""
    base = np.bincount(true, minlength=NEV) / len(true)
    recalls, tops, lifts = {}, {}, {}
    for ev in range(1, NEV):
        m = true == ev
        if m.sum() < 3:
            continue
        h = np.bincount(codes[m], minlength=K); top = int(h.argmax())
        recalls[ev] = h[top] / m.sum(); tops[ev] = top
        in_top = codes == top
        lifts[ev] = ((true[in_top] == ev).mean() / (base[ev] + 1e-9)) if in_top.sum() else 0.0
    mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0
    distinct = len(set(tops.values())) == len(tops) and len(tops) > 0
    return nmi(true, codes), mean_recall, distinct, tops
