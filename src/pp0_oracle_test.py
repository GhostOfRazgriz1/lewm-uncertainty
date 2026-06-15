"""pp0 -- SANITY GATE for the real-physics substrate: is the true-sim MPC oracle competent at pushing the
block to the goal? The oracle is the planning upper bound AND the DAgger expert, so it must work. If it does,
the pymunk substrate is viable. Run: python src/pp0_oracle_test.py --block box --episodes 12
"""
import argparse
import time
import numpy as np
from _pushphys_common import PushPhysEnv, oracle_action, scripted_push

ap = argparse.ArgumentParser()
ap.add_argument("--block", default="box"); ap.add_argument("--policy", default="scripted"); ap.add_argument("--episodes", type=int, default=12)
ap.add_argument("--T_plan", type=int, default=90); ap.add_argument("--N", type=int, default=48)
ap.add_argument("--H", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
g = np.random.default_rng(args.seed)
env = PushPhysEnv(block=args.block)

t0 = time.time(); succ = 0; steps = []
for ep in range(args.episodes):
    env.reset(g)
    for t in range(args.T_plan):
        a = scripted_push(env) if args.policy == "scripted" else oracle_action(env, g, N=args.N, H=args.H)
        env.step(a)
        if np.linalg.norm(env.block_xy() - env.goal) < env.pos_tol:
            succ += 1; steps.append(t + 1); break
dt = time.time() - t0
print(f"{args.policy.upper()} block={args.block}  success {succ}/{args.episodes} = {succ/args.episodes:.2f}  "
      f"mean steps-to-goal {np.mean(steps) if steps else float('nan'):.1f}  ({dt:.1f}s, {dt/args.episodes:.2f}s/ep)", flush=True)
