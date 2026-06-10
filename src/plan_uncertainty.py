"""M1.2 — uncertainty-aware CEM planning on Push-T. Does penalizing high-uncertainty plans improve
planning? CEM-MPC with cost = zscore(latent-dist-to-goal) + beta * zscore(MC-dropout rollout variance).
beta=0 is vanilla LeWM planning; beta>0 distrusts plans whose rollout the model is unsure about (the
MC-dropout uncertainty shown calibrated-to-error in M1.1). Compares mean best-reward across episodes.
Efficient: encode current+goal ONCE per decision, roll out S plans via predict (no re-encoding).
Run on Colab GPU:  python src/plan_uncertainty.py"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # registers swm/PushT-v1
from torchvision import transforms as TT

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm

FS, HS, HORIZON = 5, 3, 6                                          # frameskip, history, plan horizon (model steps)
S, CEM_ITERS, ELITE, MC = 96, 3, 12, 6
ACTION_SCALE = 2.0                                                 # env [-1,1] -> model's z-scored input (from plan_diagnose: CEM beats random at ~2x)
EPISODES, BUDGET = 20, 15
BETAS = [0.0, 1.0]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


@torch.no_grad()
def encode_one(pix):                                              # [3,224,224] -> [192]
    return model.encode({"pixels": pix[None, None]})["emb"][0, 0]


@torch.no_grad()
def rollout_final(cur_emb, plans, dropout):
    """cur_emb [192], plans [S,P,10] -> final predicted emb [S,192] (autoregressive, history=growing, last HS)."""
    if dropout:
        set_drop(True)
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()          # [S,1,192]
    act = model.action_encoder(plans * ACTION_SCALE)             # scale env actions -> model's trained range
    for t in range(P):
        e_tr = emb[:, -HS:]
        a_tr = act[:, :t + 1][:, -HS:]
        emb = torch.cat([emb, model.predict(e_tr, a_tr)[:, -1:]], dim=1)
    if dropout:
        set_drop(False)
    return emb[:, -1]                                            # [S,192]


@torch.no_grad()
def plan_costs(cur_emb, goal_emb, plans, beta):
    base = (rollout_final(cur_emb, plans, False) - goal_emb).pow(2).sum(-1)   # [S] latent dist to goal
    if beta == 0:
        return base
    samples = torch.stack([rollout_final(cur_emb, plans, True) for _ in range(MC)])   # [MC,S,192]
    unc = samples.var(0).sum(-1)                                 # [S] MC-dropout rollout variance
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    return z(base) + beta * z(unc)                              # scale-free combination


@torch.no_grad()
def cem(cur_emb, goal_emb, beta, gen):
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        noise = torch.randn(S, HORIZON, 10, generator=gen, device=device)
        plans = (mu + sigma * noise).clamp(-1, 1)
        elite = plans[plan_costs(cur_emb, goal_emb, plans, beta).argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)                                       # [HORIZON,10]


def run_arm(beta):
    g = torch.Generator(device=device).manual_seed(0)
    best_rewards = []
    for ep in range(EPISODES):
        env = gym.make("swm/PushT-v1", render_mode="rgb_array")
        _, info = env.reset(seed=ep)
        goal_emb = encode_one(prep(info["goal"]).to(device))
        best_r = -1e18
        for step in range(BUDGET):
            cur_emb = encode_one(prep(env.render()).to(device))
            mstep = cem(cur_emb, goal_emb, beta, g)[0].cpu().numpy()   # first model-step = 5 env actions
            done = False
            for j in range(FS):
                _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
                best_r = max(best_r, float(r))
                done = term or trunc
                if done:
                    break
            if done:
                break
        best_rewards.append(best_r)
        env.close()
    return np.array(best_rewards)


print(f"==== uncertainty-aware CEM on Push-T ({EPISODES} eps/arm, horizon {HORIZON}, S {S}) ====", flush=True)
results = {}
for beta in BETAS:
    rew = run_arm(beta); results[beta] = rew
    tag = "vanilla" if beta == 0 else f"beta={beta}"
    print(f"[{tag:10s}] mean best-reward {rew.mean():.2f} +/- {rew.std():.2f}  | per-ep {np.round(rew,1).tolist()}", flush=True)
base = results[0.0].mean()
print("\nUNCERTAINTY HELPS if a beta>0 arm's mean best-reward > vanilla's. Deltas vs vanilla:")
for beta in BETAS:
    if beta > 0:
        print(f"  beta={beta}: {results[beta].mean() - base:+.2f}")
