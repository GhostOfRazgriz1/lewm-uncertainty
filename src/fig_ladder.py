"""Figure 5 -- the evidence ladder. Per substrate (increasing realism), planning success for three
controllers: model-based (learned-model CEM), model-free actionable (event-conditioned controller + DAgger),
and the oracle (competent expert). Numbers are from the committed runs (EventEnv C3/C5/C7, PushEnv P3/P5,
PushPhysEnv pp3). Bars = representative value, whiskers = seed min/max. Run: python src/fig_ladder.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

subs = ["EventEnv\n(symbolic)", "PushEnv\n(structured contact)", "PushPhysEnv\n(real 2D physics)"]
# (value, lo, hi) per substrate
MB = [(0.13, 0.10, 0.15), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00)]      # model-based: learned-model CEM
MF = [(0.99, 0.90, 1.00), (0.60, 0.60, 0.60), (0.90, 0.85, 0.95)]      # model-free: actionable + DAgger
OR = [(0.85, 0.80, 0.95), (0.90, 0.85, 0.95), (0.93, 0.90, 0.95)]      # oracle: competent expert

x = np.arange(len(subs)); w = 0.26
fig, ax = plt.subplots(figsize=(8.4, 4.8))


def bars(data, off, color, label, hatch=None):
    v = np.array([d[0] for d in data]); lo = np.array([d[1] for d in data]); hi = np.array([d[2] for d in data])
    yerr = np.vstack([v - lo, hi - v])
    ax.bar(x + off, v, w, color=color, label=label, yerr=yerr, capsize=4,
           error_kw=dict(lw=1.2, ecolor="#333"), hatch=hatch, edgecolor="white", linewidth=0.6)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.025, f"{vi:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")


bars(MB, -w, "#7f8c8d", "model-based  (learned-model CEM)")
bars(MF, 0.0, "#c0392b", "model-free  (actionable event ctrl + DAgger)")
bars(OR, +w, "#16a085", "oracle  (competent expert)")

ax.annotate("→ 0.80–0.88\nwith more DAgger\ncoverage (P5)", xy=(1.0, 0.60), xytext=(1.32, 0.74),
            fontsize=7.5, ha="left", va="center", color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
ax.set_xticks(x); ax.set_xticklabels(subs, fontsize=10)
ax.set_ylabel("planning success", fontsize=11); ax.set_ylim(0, 1.08)
ax.set_title("Predictive events are not plannable events — actionability wins across substrates\n"
             "model-based control collapses as contact realism rises; model-free tracks the oracle",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5, frameon=False)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("docs/fig_ladder.png", dpi=130, bbox_inches="tight")
print("saved docs/fig_ladder.png")
