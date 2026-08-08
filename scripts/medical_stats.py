"""Shared paired statistics for the medical debate analysis.

Pure functions, no I/O and no API calls, so both
`aggregate_medical_results.py` and `compute_paired_tests.py` can use exactly
the same maths. Keeping them here is the point: PGR confidence intervals used
to be computed one way by the aggregator (a normal approximation) and another
way by hand for the curated bundle, so the repo shipped two different CIs for
the same quantity.

Everything is **paired at the patient-case level**. Blind, debate, and oracle
scores for one case are always resampled or permuted together, because the
three arms are measured on the same cases. Scores are the swap-averaged
per-case values produced by `aggregate_medical_results._swap_average`, so each
one is 0.0, 0.5, or 1.0.

All randomised procedures take an explicit seed and default to a fixed one, so
repeated runs of the pipeline reproduce the published numbers exactly.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Resample counts. 20k bootstrap draws put the Monte-Carlo error on a
# percentile bound well below the 0.5pp granularity of a 100-case swap-averaged
# score; 200k sign-flips give a permutation p-value floor of 1/200001 ≈ 5e-06,
# which is small enough to report the strongly-significant arms without
# claiming more resolution than the resampling supports.
BOOTSTRAP_N = 20_000
PERMUTATION_N = 200_000

# Fixed so the curated bundles are byte-reproducible across re-runs. Shares the
# value used for the deterministic DDXPlus pilot slice, purely for consistency.
DEFAULT_SEED = 20260504


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop case positions where either arm is missing.

    Pairing is only meaningful where both arms scored the same case, so a NaN
    on either side removes the whole pair rather than being filled in.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must be aligned, got {a.shape} vs {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def mean_ci(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """(mean, lo, hi) for one arm via percentile bootstrap over cases."""
    values = np.asarray([v for v in np.asarray(values, dtype=float) if not np.isnan(v)])
    if values.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_mean_diff_ci(
    treatment: np.ndarray,
    reference: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """(mean_diff, lo, hi) for `treatment - reference`, case-paired bootstrap.

    One bootstrap replicate draws case *positions* with replacement and reads
    both arms at those positions, so the correlation between the two arms is
    preserved.
    """
    t, r = _clean_pair(treatment, reference)
    if t.size == 0:
        return (float("nan"),) * 3
    diffs = t - r
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return (
        float(diffs.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_t_test(treatment: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(t_statistic, two-sided p) from a paired t-test on per-case differences.

    Reported alongside the permutation test rather than instead of it: the
    per-case differences only take values in {-1, -0.5, 0, 0.5, 1}, so they are
    not normal, and the t-test is an approximation even at n=100.
    """
    t, r = _clean_pair(treatment, reference)
    if t.size < 2:
        return float("nan"), float("nan")
    diffs = t - r
    if np.allclose(diffs, 0.0):
        # scipy returns NaN for an all-zero difference vector; 0 difference
        # with p=1 is the honest reading.
        return 0.0, 1.0
    result = stats.ttest_rel(t, r)
    return float(result.statistic), float(result.pvalue)


def paired_permutation_p(
    treatment: np.ndarray,
    reference: np.ndarray,
    n_perm: int = PERMUTATION_N,
    seed: int = DEFAULT_SEED,
    block: int = 20_000,
) -> float:
    """Two-sided paired sign-flip permutation p-value.

    Under the null that debate makes no difference to a given case, the sign of
    that case's difference is exchangeable. Each replicate flips the sign of
    every nonzero difference independently and recomputes the mean; the p-value
    is the share of replicates whose |mean| reaches the observed |mean|.

    Uses the (hits + 1) / (n_perm + 1) estimator, which never returns 0 — with
    200k replicates the floor is ~5e-06, and reporting that floor is more
    honest than reporting p = 0.

    Sign flips are generated in blocks to keep peak memory bounded; with
    n_perm=200k and 100 cases a single array would be ~160MB.
    """
    t, r = _clean_pair(treatment, reference)
    if t.size == 0:
        return float("nan")
    diffs = t - r
    n = diffs.size
    observed = abs(diffs.mean())
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    hits = 0
    remaining = n_perm
    while remaining > 0:
        size = min(block, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, nonzero.size))
        # Zero-valued differences contribute nothing to the sum under any
        # flip, but they still count toward n in the mean.
        perm_means = (signs * nonzero).sum(axis=1) / n
        hits += int(np.count_nonzero(np.abs(perm_means) >= observed - 1e-12))
        remaining -= size
    return (hits + 1) / (n_perm + 1)


def pgr(debate_acc: float, blind_acc: float, oracle_acc: float) -> float:
    """Performance Gap Recovered. NaN when the blind→oracle gap is not positive.

    A non-positive denominator means the arm has no measurable headroom for
    debate to recover, so the ratio is undefined rather than large.
    """
    gap = oracle_acc - blind_acc
    if not np.isfinite(gap) or gap <= 0:
        return float("nan")
    return (debate_acc - blind_acc) / gap


def pgr_ci(
    debate: np.ndarray,
    blind: np.ndarray,
    oracle: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """(pgr, lo, hi) via case-paired bootstrap over all three arms.

    Each replicate draws case positions once and reads debate, blind, and
    oracle at those same positions, then recomputes PGR from the three
    resampled accuracies. This matters because PGR is a ratio whose numerator
    and denominator share the blind term: resampling the arms independently
    would break that dependence and misstate the interval.

    Replicates whose bootstrap denominator collapses to <= 0 yield an
    undefined ratio and are dropped from the percentile calculation; the count
    is available via `pgr_ci_diagnostics` when that matters.
    """
    point, lo, hi, _ = pgr_ci_diagnostics(debate, blind, oracle, n_boot, seed)
    return point, lo, hi


def pgr_ci_diagnostics(
    debate: np.ndarray,
    blind: np.ndarray,
    oracle: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float, int]:
    """As `pgr_ci`, plus the number of dropped (undefined-denominator) draws."""
    d = np.asarray(debate, dtype=float)
    b = np.asarray(blind, dtype=float)
    o = np.asarray(oracle, dtype=float)
    if not (d.shape == b.shape == o.shape):
        raise ValueError("debate/blind/oracle arrays must be aligned by case")
    keep = ~(np.isnan(d) | np.isnan(b) | np.isnan(o))
    d, b, o = d[keep], b[keep], o[keep]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), 0

    point = pgr(float(d.mean()), float(b.mean()), float(o.mean()))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    d_means = d[idx].mean(axis=1)
    b_means = b[idx].mean(axis=1)
    o_means = o[idx].mean(axis=1)
    gaps = o_means - b_means
    valid = gaps > 0
    dropped = int(n_boot - np.count_nonzero(valid))
    if not np.any(valid):
        return point, float("nan"), float("nan"), dropped
    ratios = (d_means[valid] - b_means[valid]) / gaps[valid]
    return (
        point,
        float(np.percentile(ratios, 2.5)),
        float(np.percentile(ratios, 97.5)),
        dropped,
    )


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in input order.

    Controls the family-wise error rate across the set of tests passed in.
    Less blunt than plain Bonferroni: the smallest raw p is multiplied by m,
    the next by m-1, and so on, with a running maximum so the adjusted
    sequence stays monotone. NaN entries are excluded from the family and
    returned as NaN.
    """
    vals = list(pvalues)
    finite = [(i, p) for i, p in enumerate(vals) if p is not None and np.isfinite(p)]
    out = [float("nan")] * len(vals)
    m = len(finite)
    if m == 0:
        return out
    finite.sort(key=lambda pair: pair[1])
    running = 0.0
    for rank, (i, p) in enumerate(finite):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[i] = running
    return out
