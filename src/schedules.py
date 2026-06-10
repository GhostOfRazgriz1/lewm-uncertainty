"""Pure look-scheduling logic for temporal active sensing (M1.3) -- no torch/swm, unit-testable locally.

A "look-set" is the sorted list of model-steps at which the agent spends a real observation
(re-encodes the true frame), resetting its maintained latent to the truth. Step 0 is always a look
(the initial observation). Between looks the agent predicts forward, so latent-tracking error
accumulates within each inter-look segment and resets at the next look. Every policy here turns a
budget K (number of looks) and/or a per-step signal into a look-set; active_sense.py then scores each
look-set's tracking error so policies are compared AT MATCHED BUDGET (same number of looks).

Policies:
  fixed_interval_lookset(T, K)      -- K evenly-spaced looks (the baseline to beat).
  random_lookset(T, K, seed)        -- K looks at random steps (chance baseline).
  threshold_lookset(signal, tau)    -- look whenever a causal per-step signal crosses tau (the
                                       deployable rule; active_sense applies it ONLINE to MC-variance).
  oracle_lookset(scores, K)         -- K looks at the highest-scoring steps (non-causal ceiling:
                                       "if you knew which transitions are surprising, look there").
"""
from __future__ import annotations
import random as _random


def fixed_interval_lookset(T, K):
    """K evenly-spaced looks over steps [0, T), always including step 0. Exactly K distinct steps."""
    _check(T, K)
    return [(i * T) // K for i in range(K)]          # strictly increasing for 1 <= K <= T, starts at 0


def random_lookset(T, K, seed):
    """Step 0 plus K-1 distinct random steps in [1, T). Reproducible from seed."""
    _check(T, K)
    rng = _random.Random(seed)
    rest = rng.sample(range(1, T), K - 1) if K > 1 else []
    return sorted([0] + rest)


def threshold_lookset(signal, tau):
    """Memoryless causal rule: look at step 0, then at every step whose signal >= tau.

    signal[0] is ignored (step 0 is always a look). active_sense.py uses this exact rule online,
    feeding the MC-dropout predictive variance of each step -- variance naturally rises the longer
    the agent has been predicting since its last look, so a flat threshold spends looks where the
    model is least sure. Returned look count depends on tau (sweep tau to trace the budget axis)."""
    looks = [0]
    looks.extend(t for t in range(1, len(signal)) if signal[t] >= tau)
    return looks


def oracle_lookset(scores, K):
    """Step 0 plus the K-1 highest-scoring steps in [1, T). Non-causal upper bound on scheduling."""
    T = len(scores)
    _check(T, K)
    ranked = sorted(range(1, T), key=lambda t: scores[t], reverse=True)
    return sorted([0] + ranked[: K - 1])


def n_looks(lookset):
    return len(lookset)


def last_look_at(lookset, t):
    """Largest look-step <= t (the origin the agent predicts forward from at step t)."""
    prev = lookset[0]
    for L in lookset:
        if L <= t:
            prev = L
        else:
            break
    return prev


def _check(T, K):
    if not (1 <= K <= T):
        raise ValueError(f"need 1 <= K <= T, got K={K}, T={T}")
