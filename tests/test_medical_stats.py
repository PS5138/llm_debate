"""Regression tests for scripts.medical_stats.

These functions produce the published significance numbers in
`results/<family>_100/paired_tests.csv` and the PGR intervals in
`headline_results.csv`, so a silent change here would silently restate a
result. The cases below pin the properties that matter:

  - Holm adjustment matches a hand-worked example, including the running-max
    step that keeps the adjusted sequence monotone.
  - PGR is undefined (NaN) rather than large when the blind→oracle gap gives
    debate no headroom to recover.
  - Every randomised procedure is seeded, so repeated runs reproduce the
    committed numbers exactly.
  - Pairs are dropped, not filled, when either arm is missing a case.

Run:
    python tests/test_medical_stats.py
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import medical_stats as ms  # noqa: E402


def _check(desc, got, expected, tol=0.0):
    ok = (
        np.allclose(got, expected, atol=tol, equal_nan=True)
        if isinstance(expected, (list, tuple, float, int, np.floating))
        else got == expected
    )
    print(f"{'ok   ' if ok else 'FAIL '} [{desc}] got={got} expected={expected}")
    return ok


def main() -> int:
    results = []

    # ---- Holm ------------------------------------------------------------
    # sorted 0.005, 0.01, 0.03, 0.04 with m=4:
    #   0.005*4 = 0.020
    #   0.010*3 = 0.030
    #   0.030*2 = 0.060
    #   0.040*1 = 0.040 -> running max keeps 0.060
    results.append(
        _check(
            "holm-hand-worked",
            [round(x, 10) for x in ms.holm([0.01, 0.04, 0.03, 0.005])],
            [0.03, 0.06, 0.06, 0.02],
            tol=1e-12,
        )
    )
    results.append(
        _check("holm-clamps-at-one", ms.holm([0.9, 0.95]), [1.0, 1.0], tol=1e-12)
    )
    results.append(
        _check(
            "holm-skips-nan",
            ms.holm([0.01, float("nan"), 0.5]),
            [0.02, float("nan"), 0.5],
            tol=1e-12,
        )
    )
    results.append(_check("holm-empty", ms.holm([]), []))

    # ---- PGR -------------------------------------------------------------
    results.append(_check("pgr-normal", ms.pgr(0.84, 0.605, 0.84), 1.0, tol=1e-12))
    results.append(
        _check("pgr-above-oracle", ms.pgr(0.875, 0.605, 0.84), 0.27 / 0.235, tol=1e-12)
    )
    # No headroom: the denominator is the thing debate is supposed to recover,
    # so a zero or negative gap makes the ratio meaningless, not big.
    results.append(_check("pgr-no-headroom", ms.pgr(0.9, 0.6, 0.6), float("nan")))
    results.append(
        _check("pgr-oracle-below-blind", ms.pgr(0.9, 0.7, 0.6), float("nan"))
    )

    # ---- Permutation -----------------------------------------------------
    # Identical arms have no nonzero differences to flip.
    same = np.array([1.0, 0.5, 0.0, 1.0] * 25)
    results.append(
        _check(
            "perm-identical-arms",
            ms.paired_permutation_p(same, same, n_perm=2000),
            1.0,
            tol=1e-12,
        )
    )
    # A uniform +0.5 shift on every case is as extreme as the sign-flip null
    # can get, so it must land on the (hits+1)/(n+1) floor.
    shifted = np.clip(same + 0.5, 0.0, 1.0)
    p_shift = ms.paired_permutation_p(shifted, same, n_perm=2000)
    results.append(_check("perm-strong-shift-hits-floor", p_shift, 1 / 2001, tol=1e-12))

    # ---- Pairing hygiene -------------------------------------------------
    a = np.array([1.0, 0.5, np.nan, 1.0])
    b = np.array([0.0, 0.5, 1.0, np.nan])
    # Only positions 0 and 1 are valid: diffs are +1.0 and 0.0 -> mean 0.5.
    results.append(
        _check("nan-pairs-dropped", ms.paired_mean_diff_ci(a, b)[0], 0.5, tol=1e-12)
    )
    try:
        ms.paired_mean_diff_ci(np.array([1.0, 0.0]), np.array([1.0]))
        results.append(_check("misaligned-arrays-raise", "no raise", "ValueError"))
    except ValueError:
        results.append(_check("misaligned-arrays-raise", "ValueError", "ValueError"))

    # ---- Determinism -----------------------------------------------------
    # The published bundle must be reproducible from the same cached CSVs.
    rng = np.random.default_rng(20260504)
    d = rng.choice([0.0, 0.5, 1.0], 100)
    bl = rng.choice([0.0, 0.5, 1.0], 100)
    orc = np.clip(d + 0.5, 0.0, 1.0)
    for name, fn in (
        ("mean_ci", lambda: ms.mean_ci(d)),
        ("paired_mean_diff_ci", lambda: ms.paired_mean_diff_ci(d, bl)),
        ("pgr_ci", lambda: ms.pgr_ci(d, bl, orc)),
        ("paired_permutation_p", lambda: ms.paired_permutation_p(d, bl, n_perm=2000)),
    ):
        first, second = fn(), fn()
        results.append(_check(f"deterministic-{name}", first, second, tol=0.0))

    # ---- Empty input -----------------------------------------------------
    empty = np.array([])
    results.append(_check("mean_ci-empty", ms.mean_ci(empty)[0], float("nan")))
    results.append(
        _check(
            "paired_mean_diff_ci-empty",
            ms.paired_mean_diff_ci(empty, empty)[0],
            float("nan"),
        )
    )

    print()
    failures = len([r for r in results if not r])
    if failures:
        print(f"{failures} regression(s) detected.")
        return 1
    print(f"{len(results)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
