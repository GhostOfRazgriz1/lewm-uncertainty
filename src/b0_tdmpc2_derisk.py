"""B0 -- DE-RISK GATE for Direction B: a free uncertainty MONITOR on a COMPETENT world model.

The whole de-confounding plan (re-run M1.2/A2 control + M1.6/M2.2 monitoring on a planner that can
ACTUALLY plan, unlike LeWM) rests on two facts that must hold in OUR Colab setup BEFORE any paper
framing commits. This script is the go/no-go gate that checks both, cheaply, with zero training:

  GATE 1 (competence): a PRETRAINED TD-MPC2 single-task checkpoint reproduces a competent return in
      our setup -- mean episode return >> random-action floor (and in the published ballpark). If this
      fails it's a deps/runtime problem, not a research signal. This is the OGBench-Cube risk; we want
      it to fail HERE (an afternoon) not three weeks into a paper.

  GATE 2 (live signal): there is an uncertainty signal to monitor WITH. We read two free signals off
      the frozen model at every state: (a) Q-ENSEMBLE disagreement (TD-MPC2 ships num_q=5 value heads),
      (b) ONE-STEP latent dynamics error ||next(z,a) - encode(obs')||. GATE 2 passes if at least one
      signal is (i) non-degenerate (varies across states) AND (ii) correlates with the realized
      one-step model error (Spearman). If the Q-ensemble is collapsed/flat (heads trained jointly ->
      may agree), the fallback is the M2.1 recipe: train K action-free forward heads on the frozen
      latent -- noted, not run here.

NOTE the SimNorm caveat: TD-MPC2's latent is simplicially normalized, NOT Gaussian, so the LeJEPA
SHELL/OOD signal does NOT port here. On this substrate the monitor is ENSEMBLE disagreement (our
sharpest signal anyway, M2.1/M2.2). The shell facet stays on the JEPA side (cross-substrate breadth).

GO  = GATE1 and GATE2 pass -> build E1 (monitor under shift) + E2 (uncertainty-cost MPC, the swing).
NO-GO(1) = checkpoint not competent -> deps/runtime triage (see spec).
NO-GO(2) = no live signal -> fall back to trained action-free forward heads before E1/E2.

Colab setup (runtime=GPU) in docs/B0-tdmpc2-derisk-spec.md.
Run:  python src/b0_tdmpc2_derisk.py --task cheetah-run --seed 1 --episodes 20
"""
import os
import sys
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

# --- args ----------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="cheetah-run", help="DMControl single-task name (e.g. cheetah-run, cup-catch, walker-walk)")
ap.add_argument("--seed", type=int, default=1, help="checkpoint seed in {1,2,3}")
ap.add_argument("--episodes", type=int, default=20, help="eval episodes for the checkpoint")
ap.add_argument("--random_episodes", type=int, default=10, help="random-action floor episodes")
ap.add_argument("--tdmpc2_dir", default="/content/tdmpc2/tdmpc2", help="path to the tdmpc2/tdmpc2 package dir")
args = ap.parse_args()

assert torch.cuda.is_available(), "TD-MPC2 eval needs a GPU runtime (Colab: Runtime > Change runtime type > GPU)."
os.environ.setdefault("MUJOCO_GL", "egl")
device = "cuda"

# --- import the tdmpc2 package (its modules use bare 'from common import ...', so cd + path it) ----
TDMPC2_DIR = args.tdmpc2_dir
assert os.path.isdir(TDMPC2_DIR), f"tdmpc2 package dir not found: {TDMPC2_DIR} (clone the repo, see spec)."
sys.path.insert(0, TDMPC2_DIR)
os.chdir(TDMPC2_DIR)                                              # hydra config_path + bare imports expect cwd here
from hydra import compose, initialize_config_dir                 # noqa: E402
from common.parser import parse_cfg                              # noqa: E402
from common.seed import set_seed                                 # noqa: E402
from common import math as tdmath                                # two_hot_inv lives here  # noqa: E402
from envs import make_env                                        # noqa: E402
from tdmpc2 import TDMPC2                                         # noqa: E402
from huggingface_hub import hf_hub_download                      # noqa: E402

# --- approx published DMControl returns (compare to repo results/; rough, not load-bearing) -------
REF = {"cheetah-run": 850, "cup-catch": 980, "walker-walk": 975, "walker-run": 820, "finger-spin": 980,
       "reacher-easy": 980, "cartpole-swingup": 870, "acrobot-swingup": 420, "fish-swim": 800}

# --- checkpoint --------------------------------------------------------------------------------
ckpt = hf_hub_download("nicklashansen/tdmpc2", filename=f"dmcontrol/{args.task}-{args.seed}.pt")
print(f"[ckpt] {ckpt}", flush=True)

# --- build cfg exactly like evaluate.py: parse_cfg -> make_env (mutates dims) -> TDMPC2 -> load ---
import hydra.utils as _hu                                         # the compose() API has no Hydra runtime,
_hu.get_original_cwd = lambda: os.getcwd()                       # so parse_cfg's get_original_cwd() (work_dir, unused here) errors -> patch it
with initialize_config_dir(config_dir=TDMPC2_DIR, version_base=None):
    cfg = compose(config_name="config", overrides=[
        f"task={args.task}", f"checkpoint={ckpt}", "model_size=5",
        "compile=false", f"eval_episodes={args.episodes}", "save_video=false", "seed=1"])
cfg = parse_cfg(cfg)
set_seed(cfg.seed)
env = make_env(cfg)
agent = TDMPC2(cfg)
agent.load(cfg.checkpoint)
agent.model.eval()
print(f"[cfg] task={cfg.task} obs={cfg.obs} num_q={cfg.num_q} latent_dim={cfg.latent_dim} action_dim={cfg.action_dim}", flush=True)


@torch.no_grad()
def enc(obs):
    return agent.model.encode(obs.to(device).unsqueeze(0), None)              # [1, latent_dim]


@torch.no_grad()
def signals(z, a):
    """Q-ensemble disagreement + predicted next latent, off the frozen model."""
    a = a.to(device).unsqueeze(0)
    q_logits = agent.model.Q(z, a, None, return_type="all")                   # [num_q, 1, num_bins]
    q_vals = tdmath.two_hot_inv(q_logits, cfg).squeeze(-1).squeeze(-1)        # [num_q, 1] -> per-head scalar
    q_disag = q_vals.var(dim=0).item()                                       # disagreement across value heads
    z_pred = agent.model.next(z, a, None)                                    # [1, latent_dim]
    return q_disag, z_pred


# --- competent rollouts with signal dump ---------------------------------------------------------
rec = {"q_disag": [], "onestep": [], "znorm": [], "qmean": []}
returns, successes = [], []
for ep in range(cfg.eval_episodes):
    obs, done, ep_r, t = env.reset(), False, 0.0, 0
    while not done:
        z = enc(obs)
        a = agent.act(obs, t0=(t == 0), task=None)                           # mirrors evaluate.py (no eval_mode kwarg)
        q_disag, z_pred = signals(z, a)
        obs2, reward, done, info = env.step(a)
        with torch.no_grad():
            onestep = (z_pred - enc(obs2)).norm().item()                     # realized one-step latent error
        rec["q_disag"].append(q_disag); rec["onestep"].append(onestep)
        rec["znorm"].append(z.norm().item())
        rec["qmean"].append(float(tdmath.two_hot_inv(agent.model.Q(z, a.to(device).unsqueeze(0), None, return_type="all"), cfg).mean()))
        ep_r += float(reward); t += 1
        obs = obs2
    returns.append(ep_r); successes.append(float(info.get("success", np.nan)))
    print(f"  ep {ep:2d}: return {ep_r:7.1f}  len {t}", flush=True)

# --- random-action floor -------------------------------------------------------------------------
rand_returns = []
for ep in range(args.random_episodes):
    obs, done, ep_r = env.reset(), False, 0.0
    while not done:
        obs, reward, done, info = env.step(env.rand_act())
        ep_r += float(reward)
    rand_returns.append(ep_r)

for k in rec:
    rec[k] = np.asarray(rec[k], dtype="float64")
R, Rstd = float(np.mean(returns)), float(np.std(returns))
F, Fstd = float(np.mean(rand_returns)), float(np.std(rand_returns))

# --- diagnostics ---------------------------------------------------------------------------------
def cv(x):
    return float(np.std(x) / (abs(np.mean(x)) + 1e-9))


sp_q = spearmanr(rec["q_disag"], rec["onestep"]).correlation if rec["q_disag"].std() > 0 else 0.0
sp_self = spearmanr(rec["onestep"][:-1], rec["onestep"][1:]).correlation     # autocorr sanity of the target
ref = REF.get(args.task)

print("\n==== B0 de-risk gate :", args.task, f"(seed {args.seed}) ====")
print(f"  GATE 1  competence")
print(f"    checkpoint return : {R:8.1f} +/- {Rstd:.1f}   ({cfg.eval_episodes} eps)")
print(f"    random floor      : {F:8.1f} +/- {Fstd:.1f}   ({args.random_episodes} eps)")
print(f"    published approx  : {ref if ref else 'n/a'}   (compare to repo results/ for exact)")
margin = R - F
gate1 = (R > 600) and (margin > 5 * max(1.0, abs(F))) and (R > F + 3 * (Rstd + Fstd))
print(f"    margin over random: {margin:8.1f}   ->  GATE1 {'PASS' if gate1 else 'FAIL'}")

print(f"  GATE 2  live uncertainty signal")
print(f"    Q-ensemble disag  : mean {rec['q_disag'].mean():.4g}  CV {cv(rec['q_disag']):.3f}  (need CV>0.1: {'ok' if cv(rec['q_disag'])>0.1 else 'FLAT'})")
print(f"    one-step lat. err : mean {rec['onestep'].mean():.4g}  CV {cv(rec['onestep']):.3f}   (target signal; autocorr {sp_self:+.2f})")
print(f"    Spearman(q_disag, one-step err): {sp_q:+.3f}   (need |.|>0.15 for a usable monitor)")
gate2 = (cv(rec["q_disag"]) > 0.1) and (abs(sp_q) > 0.15)
print(f"    ->  GATE2 {'PASS' if gate2 else 'FAIL (Q-ensemble flat/uninformative -> fall back to trained action-free heads, M2.1 recipe)'}")

go = gate1 and gate2
print("\n  VERDICT:", "GO -- build E1 (monitor under shift) + E2 (uncertainty-cost MPC)." if go else
      ("NO-GO(1): checkpoint not competent in this setup -> deps/runtime triage (spec)." if not gate1 else
       "NO-GO(2): no free live signal -> train K action-free forward heads on frozen latent first, then re-gate."))

# --- cache + figure ------------------------------------------------------------------------------
out = f"/content/b0_{args.task}_s{args.seed}"
torch.save({"task": args.task, "seed": args.seed, "returns": returns, "rand_returns": rand_returns,
            "records": {k: rec[k] for k in rec}, "gate1": gate1, "gate2": gate2, "go": go,
            "spearman_q_onestep": sp_q}, out + "_records.pt")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].bar(["checkpoint", "random"], [R, F], yerr=[Rstd, Fstd], color=["#27ae60", "#999"], capsize=5)
if ref:
    ax[0].axhline(ref, ls="--", color="#c0392b", label=f"published ~{ref}"); ax[0].legend(fontsize=8)
ax[0].set_ylabel("episode return"); ax[0].set_title(f"GATE 1: competence ({args.task})")
ax[1].hist(rec["q_disag"] / (rec["q_disag"].mean() + 1e-9), bins=40, alpha=.6, color="#8e44ad", label="Q-ensemble disag (norm)")
ax[1].hist(rec["onestep"] / (rec["onestep"].mean() + 1e-9), bins=40, alpha=.5, color="#2980b9", label="one-step err (norm)")
ax[1].set_title("GATE 2: signal spread (flat=dead)"); ax[1].legend(fontsize=8); ax[1].set_xlabel("value / mean")
ax[2].scatter(rec["q_disag"], rec["onestep"], s=6, alpha=.3, color="#8e44ad")
ax[2].set_xlabel("Q-ensemble disagreement"); ax[2].set_ylabel("one-step latent error")
ax[2].set_title(f"GATE 2: signal vs error  (Spearman {sp_q:+.2f})")
for a in ax:
    a.grid(alpha=.3)
fig.suptitle(f"B0 de-risk gate -- TD-MPC2 {args.task}  |  VERDICT: {'GO' if go else 'NO-GO'}", fontweight="bold")
fig.tight_layout(); fig.savefig(out + ".png", dpi=110)
print(f"\nsaved {out}.png  and  {out}_records.pt")
