"""(3) INFRA DE-RISK: can a DMC (or other) pixel control env render on Colab?

We got burned twice on Colab mujoco/OGBench rendering (OpenGL ctx -> segfault -> numpy conflict). Before
building a JEPA world model on DMC, confirm pixels render at all. Tries several backends and substrates and
reports which work, so we pick the substrate for experiment (3) with eyes open.

Run on Colab GPU FIRST (cheap):  python src/dmc_smoke.py
Colab install (run in a cell before this):
  !pip install -q dm_control shimmy[dm-control] gymnasium ale-py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")                        # the Cube fix; egl = headless GL
import numpy as np

results = {}


def shape_or(x):
    a = np.asarray(x)
    return f"OK pixels {a.shape} dtype {a.dtype} range [{a.min()},{a.max()}]"


# --- 1) dm_control suite, direct pixel render (the credible WIMLE-style substrate) ----------------
try:
    from dm_control import suite                                  # noqa
    env = suite.load("cartpole", "swingup")
    env.reset()
    pix = env.physics.render(84, 84, camera_id=0)
    results["dm_control.suite (egl)"] = shape_or(pix)
except Exception as e:
    results["dm_control.suite (egl)"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"

# --- 2) dm_control via gymnasium/shimmy (the API we'd actually use) -------------------------------
try:
    import gymnasium as gym                                       # noqa
    env = gym.make("dm_control/cartpole-swingup-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    results["gymnasium dm_control/ (shimmy)"] = shape_or(frame)
    env.close()
except Exception as e:
    results["gymnasium dm_control/ (shimmy)"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"

# --- 3) gymnasium mujoco classic (HalfCheetah pixels) ---------------------------------------------
try:
    import gymnasium as gym                                       # noqa
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    results["gymnasium HalfCheetah-v5"] = shape_or(frame)
    env.close()
except Exception as e:
    results["gymnasium HalfCheetah-v5"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"

# --- 4) Atari/ALE (NO mujoco GL -- the low-infra-risk fallback substrate) -------------------------
try:
    import gymnasium as gym                                       # noqa
    import ale_py                                                 # noqa
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    results["ALE/Pong-v5 (no mujoco)"] = shape_or(frame)
    env.close()
except Exception as e:
    results["ALE/Pong-v5 (no mujoco)"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"

# --- report ---------------------------------------------------------------------------------------
print("\n==== (3) pixel-rendering smoke test on Colab ====")
for k, v in results.items():
    print(f"  {k:34}: {v}")
ok = [k for k, v in results.items() if v.startswith("OK")]
print("\n  WORKING substrates:", ok if ok else "NONE")
if any("dm_control" in k or "HalfCheetah" in k for k in ok):
    print("  => DMC/mujoco pixels render -> the credible WIMLE-style substrate for (3) is viable. Scope the build.")
elif "ALE/Pong-v5 (no mujoco)" in ok:
    print("  => mujoco blocked again, but ALE renders -> fall back to Atari pixels for (3) (no GL rabbit hole).")
else:
    print("  => NOTHING renders -> build a custom matplotlib/numpy pixel control env (point-mass reacher) for (3).")
