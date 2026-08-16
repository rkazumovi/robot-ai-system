"""
drift.py — Model drift detection for the deployed CV model.

Tracks a rolling window of recent CV model outputs (confidence scores,
defect pixel fractions) and compares their distribution against a
reference baseline (e.g. captured from the validation set at deployment
time) using the two-sample Kolmogorov-Smirnov test -- a standard,
well-established way to detect whether two samples come from different
distributions, without assuming any particular distribution shape.

A significant KS-test result (low p-value) suggests the live input
distribution has shifted from what the model was trained/validated on
-- e.g. a camera angle change, lighting change, or genuinely novel
defect types appearing -- which is exactly the kind of silent failure
mode a production CV system needs to catch, since the model won't
necessarily produce obviously wrong-looking outputs even when its
inputs have drifted outside its training distribution.

Run this file directly to verify the KS-test correctly distinguishes
"no drift" (same distribution) from genuine drift (shifted distribution),
using synthetic data with known ground truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

@dataclass
class DriftMonitor:
    """
    Maintains a rolling window of recent observations for one metric
    (e.g. confidence scores) and tests for drift against a fixed
    reference baseline.

    Args:
        reference: baseline sample (e.g. validation-set confidence
            scores, captured once at deployment time).
        window_size: how many recent live observations to compare
            against the reference.
        alpha: significance threshold for the KS test. p < alpha is
            flagged as drift. 0.01 (not the more common 0.05) is used
            by default to reduce false alarms from a single noisy batch
            -- drift monitoring should flag persistent shifts, not
            every statistical fluctuation.
    """

    reference: np.ndarray
    window_size: int = 200
    alpha: float = 0.01
    _window: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self):
        self._window = deque(maxlen=self.window_size)

    def add_observation(self, value: float) -> None:
        self._window.append(value)

    def check_drift(self) -> dict:
        """
        Runs a two-sample KS test comparing the current rolling window
        against the reference distribution.

        Returns:
            dict with 'n_observations', 'ks_statistic', 'p_value',
            'is_drifted' (bool), 'window_mean', 'reference_mean'.
            If there aren't enough observations yet, is_drifted is
            False and a 'reason' field explains why.
        """
        n = len(self._window)
        if n < 30:  # KS test needs a reasonable sample size to be meaningful
            return {
                "n_observations": n,
                "is_drifted": False,
                "reason": f"Insufficient observations ({n} < 30) for a reliable test.",
            }

        window_array = np.array(self._window)
        ks_statistic, p_value = stats.ks_2samp(self.reference, window_array)

        return {
            "n_observations": n,
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "is_drifted": bool(p_value < self.alpha),
            "window_mean": float(window_array.mean()),
            "reference_mean": float(self.reference.mean()),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Scenario 1: no drift (same distribution) ===")
    reference = rng.normal(loc=0.9, scale=0.05, size=500)  # e.g. typical confidence scores
    monitor = DriftMonitor(reference=reference, window_size=200, alpha=0.01)

    # Feed observations from the SAME distribution as the reference.
    for _ in range(200):
        monitor.add_observation(float(rng.normal(loc=0.9, scale=0.05)))

    result = monitor.check_drift()
    print(f"  n_observations: {result['n_observations']}")
    print(f"  ks_statistic:   {result['ks_statistic']:.4f}")
    print(f"  p_value:        {result['p_value']:.4f}")
    print(f"  window_mean:    {result['window_mean']:.4f}  reference_mean: {result['reference_mean']:.4f}")
    print(f"  is_drifted:     {result['is_drifted']} (expected False)")
    assert not result["is_drifted"], "Should NOT detect drift when distributions match."
    print("Correctly did NOT flag drift for matching distributions.")

    print("\n=== Scenario 2: genuine drift (shifted distribution) ===")
    monitor2 = DriftMonitor(reference=reference, window_size=200, alpha=0.01)

    # Feed observations from a DIFFERENT (shifted, lower-confidence) distribution --
    # simulating e.g. a camera/lighting change degrading model confidence.
    for _ in range(200):
        monitor2.add_observation(float(rng.normal(loc=0.65, scale=0.08)))

    result2 = monitor2.check_drift()
    print(f"  n_observations: {result2['n_observations']}")
    print(f"  ks_statistic:   {result2['ks_statistic']:.4f}")
    print(f"  p_value:        {result2['p_value']:.6f}")
    print(f"  window_mean:    {result2['window_mean']:.4f}  reference_mean: {result2['reference_mean']:.4f}")
    print(f"  is_drifted:     {result2['is_drifted']} (expected True)")
    assert result2["is_drifted"], "SHOULD detect drift when distributions genuinely differ."
    print("Correctly flagged drift for shifted distribution.")

    print("\n=== Scenario 3: insufficient data (fewer than 30 observations) ===")
    monitor3 = DriftMonitor(reference=reference, window_size=200, alpha=0.01)
    for _ in range(10):
        monitor3.add_observation(float(rng.normal(loc=0.9, scale=0.05)))
    result3 = monitor3.check_drift()
    print(f"  n_observations: {result3['n_observations']}")
    print(f"  is_drifted: {result3['is_drifted']} (expected False)")
    print(f"  reason: {result3.get('reason')}")
    assert not result3["is_drifted"]
    assert "Insufficient" in result3["reason"]
    print("Correctly handled insufficient-data case without false alarm.")

    print("\n=== Scenario 4: rolling window correctly bounded ===")
    monitor4 = DriftMonitor(reference=reference, window_size=50)
    for i in range(200):
        monitor4.add_observation(float(i))
    assert len(monitor4._window) == 50, "Window should not grow beyond window_size."
    print(f"  Window size after 200 additions (max 50): {len(monitor4._window)} (expected 50)")
    print("Rolling window correctly bounded, oldest observations evicted.")

    print("\nAll drift detection checks passed.")