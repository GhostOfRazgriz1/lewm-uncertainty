"""Diagnostic: is the vanilla planner even working? Compare RANDOM actions vs vanilla CEM at several
action SCALES. The model was trained on z-scored actions; raw [-1,1] is likely the wrong scale -> bad
rollouts -> CEM can't steer. The scale maps env actions [-1,1] -> the model's expected input (~1/std).
If some scale's CEM clearly beats random, planning works there and the M1.2 uncertainty test is valid.
Run on Colab GPU:  python src/plan_diagnose.py"""
import sys
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel as swm
from torchvision import transforms as TT

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm

FS, HS, HORIZON, S, CEM_ITERS, ELITE = 5, 3, 6, 96, 3, 12
EPISODES, BUDGET = 8, 15
SCALES = [1.0, 2.0, 3.0]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def encode_one(pix):
    return model.encode({"pixels": pix[None, None]})["emb"][0, 0]


@torch.no_grad()
def rollout_final(cur_emb, plans, scale):
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * scale)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, goal_emb, scale, gen):
    mu = torch.zeros(HORIZON, 10, device=device); sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sigma * torch.randn(S, HORIZON, 10, generator=gen, device=device)).clamp(-1, 1)
        cost = (rollout_final(cur_emb, plans, scale) - goal_emb).pow(2).sum(-1)
        elite = plans[cost.argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


def episode_reward(policy_fn, seed):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    _, info = env.reset(seed=seed)
    goal_emb = encode_one(prep(info["goal"]).to(device))
    best = -1e18
    for step in range(BUDGET):
        mstep = policy_fn(env, goal_emb)
        for j in range(FS):
            _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
            best = max(best, float(r))
            if term or trunc:
                env.close(); return best
    env.close(); return best


print(f"==== planner diagnostic ({EPISODES} eps) ====", flush=True)
g = torch.Generator(device=device).manual_seed(0)
rng = np.random.default_rng(0)
rand = np.array([episode_reward(lambda env, ge: rng.uniform(-1, 1, 10), ep) for ep in range(EPISODES)])
print(f"[random      ] mean best-reward {rand.mean():.2f} +/- {rand.std():.2f}", flush=True)
for sc in SCALES:
    def pol(env, ge, sc=sc):
        cur = encode_one(prep(env.render()).to(device))
        return cem(cur, ge, sc, g)[0].cpu().numpy()
    r = np.array([episode_reward(pol, ep) for ep in range(EPISODES)])
    print(f"[CEM scale={sc:<4}] mean best-reward {r.mean():.2f} +/- {r.std():.2f}  | delta vs random {r.mean()-rand.mean():+.2f}", flush=True)
print("\nPlanning WORKS at a scale if CEM mean-best-reward >> random. That scale validates M1.2; if none beats")
print("random, the model's rollouts are unreliable without the real action normalizer (needs the dataset stats).")
