"""Local unit tests for the pure look-scheduling logic (no torch/swm needed).

Budget-matching is the crux of a fair active-sensing comparison: every policy must spend the SAME
number of looks, or a lower tracking error is meaningless. These tests pin that invariant plus the
shape of each schedule. Run:  python tests/test_schedules.py   (or: pytest tests/test_schedules.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schedules import (  # noqa: E402
    fixed_interval_lookset, random_lookset, threshold_lookset, oracle_lookset,
    last_look_at, n_looks,
)


def test_fixed_interval_exact_budget_and_spacing():
    for T in (5, 12, 20, 37):
        for K in (1, 2, 3, T // 2, T):
            ls = fixed_interval_lookset(T, K)
            assert n_looks(ls) == K, (T, K, ls)             # exact budget
            assert ls[0] == 0                                # step 0 always a look
            assert ls == sorted(set(ls))                     # strictly increasing, distinct
            assert all(0 <= s < T for s in ls)
    assert fixed_interval_lookset(12, 4) == [0, 3, 6, 9]     # even spacing
    assert fixed_interval_lookset(10, 1) == [0]              # single look = just the initial obs


def test_random_exact_budget_reproducible_and_varies():
    ls = random_lookset(20, 6, seed=0)
    assert n_looks(ls) == 6 and ls[0] == 0
    assert ls == sorted(set(ls)) and all(0 <= s < 20 for s in ls)
    assert random_lookset(20, 6, seed=0) == ls               # reproducible
    assert random_lookset(20, 6, seed=1) != ls               # seed actually varies the draw


def test_threshold_rule_memoryless():
    flat = [0.0] * 10
    assert threshold_lookset(flat, tau=0.5) == [0]           # nothing crosses -> only the initial look
    spike = [0.0, 0.1, 0.1, 9.0, 0.1, 9.0, 0.1, 0.1, 0.1, 0.1]
    assert threshold_lookset(spike, tau=1.0) == [0, 3, 5]    # looks exactly where signal >= tau
    assert threshold_lookset([5.0] * 6, tau=1.0) == [0, 1, 2, 3, 4, 5]  # all above -> look every step
    # higher tau -> never more looks (monotone budget vs threshold)
    sig = [0.0, 2.0, 1.0, 3.0, 0.5, 2.5]
    assert n_looks(threshold_lookset(sig, 3.0)) <= n_looks(threshold_lookset(sig, 1.0))


def test_oracle_picks_top_scores_at_exact_budget():
    scores = [0.0, 0.2, 9.0, 0.1, 5.0, 0.3, 7.0]             # step 0 score ignored
    assert oracle_lookset(scores, K=1) == [0]
    assert oracle_lookset(scores, K=3) == [0, 2, 6]          # two highest: steps 2 (9.0) and 6 (7.0)
    assert oracle_lookset(scores, K=4) == [0, 2, 4, 6]       # + step 4 (5.0)
    assert n_looks(oracle_lookset(scores, 5)) == 5


def test_last_look_at_finds_segment_origin():
    ls = [0, 4, 9]
    assert [last_look_at(ls, t) for t in range(11)] == [0, 0, 0, 0, 4, 4, 4, 4, 4, 9, 9]


def _all_tests():
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for fn in _all_tests():
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(_all_tests())} SCHEDULE TESTS PASSED")
