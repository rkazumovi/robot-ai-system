"""
anomaly.py — FFT-based vibration anomaly detection for robot joints.

Approach
-----------
A healthy joint driven by a periodic operating torque produces a
vibration signal (joint angular velocity/acceleration) with energy
concentrated at the drive frequency and its harmonics. Mechanical faults
(bearing wear, gear defects, loose fasteners) typically introduce NEW
spectral energy at characteristic fault frequencies not present in
healthy operation -- e.g. ball-pass frequencies for bearing defects,
mesh frequencies for gear faults. This is standard practice in real
industrial vibration monitoring (ISO 10816 and similar), not something
specific to this project.

Detection here: take the FFT of a "baseline" (known-healthy) signal and
a "test" signal, and flag frequency bins where the test signal has
significantly more energy than the baseline -- i.e. new spectral content
that shouldn't be there.

Run this file directly to:
  1) Verify FFT peak-frequency detection against pure sinusoids of known
     frequency (ground-truth check).
  2) Generate healthy vs. fault-injected joint vibration signals (via
     dynamics.py's joint model) and verify the detector correctly flags
     the faulty signal and does not false-alarm on the healthy one or on
     pure noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dynamics import JointDynamicsParams, simulate_joint_trajectory  # noqa: E402

def compute_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided amplitude spectrum of `signal`, sampled at interval dt.

    Applies a Hann window to reduce spectral leakage (standard practice --
    without windowing, a signal that isn't an exact integer number of
    periods within the sampled window leaks energy into neighboring
    frequency bins, which would look like spurious anomaly signal).

    Returns:
        (freqs, magnitude): freqs in Hz (0 to Nyquist), magnitude the
        corresponding one-sided amplitude spectrum.
    """
    n = len(signal)
    window = np.hanning(n)
    # Amplitude correction factor for the Hann window (so peak amplitude
    # estimates are accurate, not just relative shape).
    window_correction = 1.0 / window.mean()

    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=dt)

    magnitude = (2.0 / n) * np.abs(spectrum) * window_correction
    magnitude[0] /= 2  # DC bin shouldn't be doubled by the one-sided 2/n scaling

    return freqs, magnitude


def find_peak_frequency(freqs: np.ndarray, magnitude: np.ndarray, f_min: float = 0.0) -> float:
    """Returns the frequency (Hz) of the largest spectral peak at or above f_min."""
    mask = freqs >= f_min
    idx = np.argmax(magnitude[mask])
    return freqs[mask][idx]


def band_average_spectrum(
    freqs: np.ndarray, magnitude: np.ndarray, band_width_hz: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Groups the spectrum into fixed-width frequency bands and averages
    magnitude within each band.

    Why this matters: comparing raw single FFT bins between two
    independent noise realizations is statistically unreliable -- each
    bin's magnitude has substantial natural variance (a bin magnitude
    for Gaussian noise is Rayleigh-distributed), so two healthy signals
    can easily show a >3x ratio in some individual bin purely by chance,
    with no real anomaly present. Averaging over a frequency band
    reduces that variance by roughly sqrt(bins_per_band), which is
    standard practice in real vibration analysis (e.g. comparing energy
    in characteristic fault-frequency bands rather than single bins),
    while still localizing a genuine fault tone that concentrates energy
    in one band.

    Returns:
        (band_centers, band_magnitudes)
    """
    f_max = freqs[-1]
    n_bands = max(int(f_max / band_width_hz), 1)
    band_edges = np.linspace(0, f_max, n_bands + 1)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2
    band_magnitudes = np.zeros(n_bands)

    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        if mask.any():
            band_magnitudes[i] = magnitude[mask].mean()

    return band_centers, band_magnitudes


def detect_spectral_anomalies(
    baseline_signal: np.ndarray,
    test_signal: np.ndarray,
    dt: float,
    threshold_ratio: float = 3.0,
    min_absolute_threshold: float = 1e-4,
    band_width_hz: float = 1.0,
) -> dict:
    """
    Compares the test signal's spectrum against a known-healthy baseline
    spectrum, using band-averaged energy (see band_average_spectrum) to
    stay robust to natural bin-to-bin noise variance, and flags bands
    with anomalous new energy.

    Args:
        baseline_signal: known-healthy reference signal.
        test_signal: signal to check for anomalies (same length/dt as baseline).
        dt: sample interval (s).
        threshold_ratio: a band is flagged if test energy exceeds
            threshold_ratio x baseline energy in that band.
        min_absolute_threshold: bands are only flagged if the ABSOLUTE
            test magnitude also exceeds this floor -- avoids flagging
            tiny, physically insignificant ratios in near-zero bins.
        band_width_hz: width of each frequency band for averaging.

    Returns:
        dict with 'freqs' (band centers), 'baseline_mag', 'test_mag',
        'anomaly_mask' (boolean array), 'anomaly_freqs' (flagged band
        centers), 'is_anomalous' (bool, True if any band flagged).
    """
    assert len(baseline_signal) == len(test_signal), "Signals must be the same length."

    freqs_raw, baseline_mag_raw = compute_spectrum(baseline_signal, dt)
    _, test_mag_raw = compute_spectrum(test_signal, dt)

    freqs, baseline_mag = band_average_spectrum(freqs_raw, baseline_mag_raw, band_width_hz)
    _, test_mag = band_average_spectrum(freqs_raw, test_mag_raw, band_width_hz)

    ratio = test_mag / (baseline_mag + 1e-12)
    raw_mask = (ratio > threshold_ratio) & (test_mag > min_absolute_threshold)

    # Require at least 2 ADJACENT flagged bands to declare a real anomaly.
    # A genuine fault tone spreads energy into neighboring bands (spectral
    # leakage even after windowing, as seen in our own fault-detection
    # test), whereas an isolated single flagged band is much more likely
    # to be a statistical fluke from natural noise variance -- comparing
    # hundreds of independent bands, a handful of chance exceedances are
    # expected even with no real anomaly present (a standard multiple-
    # comparisons effect). This clustering requirement is a simple,
    # physically-motivated filter for that, not an arbitrary threshold tweak.
    anomaly_mask = np.zeros_like(raw_mask)
    for i in range(len(raw_mask)):
        if raw_mask[i] and (
            (i > 0 and raw_mask[i - 1]) or (i < len(raw_mask) - 1 and raw_mask[i + 1])
        ):
            anomaly_mask[i] = True

    return {
        "freqs": freqs,
        "baseline_mag": baseline_mag,
        "test_mag": test_mag,
        "anomaly_mask": anomaly_mask,
        "anomaly_freqs": freqs[anomaly_mask],
        "is_anomalous": bool(anomaly_mask.any()),
    }


def simulate_vibration_signal(
    drive_freq_hz: float,
    duration: float,
    dt: float = 0.001,
    fault_freq_hz: float | None = None,
    fault_amplitude: float = 0.0,
) -> dict:
    """
    Simulates a single joint's vibration signal via the damped-pendulum
    joint dynamics model (dynamics.py), driven by a periodic torque at
    drive_freq_hz. Optionally injects an extra harmonic at fault_freq_hz
    to simulate a mechanical fault (e.g. bearing defect frequency).

    Returns dict with 't', 'theta_dot' (used as the vibration signal --
    angular velocity is the standard vibration proxy for rotary joints).
    """
    params = JointDynamicsParams(
        inertia=np.array([0.05]),
        damping=np.array([0.05]),
        gravity_coeff=np.array([0.0]),  # horizontal joint, isolate vibration behavior
    )

    omega_drive = 2 * np.pi * drive_freq_hz
    omega_fault = 2 * np.pi * fault_freq_hz if fault_freq_hz is not None else 0.0

    def torque_fn(t):
        tau = 0.1 * np.sin(omega_drive * t)
        if fault_freq_hz is not None:
            tau = tau + fault_amplitude * np.sin(omega_fault * t)
        return np.array([tau])

    result = simulate_joint_trajectory(
        theta0=np.array([0.0]), theta_dot0=np.array([0.0]),
        params=params, torque_fn=torque_fn, duration=duration, dt=dt,
    )
    return {"t": result["t"], "theta_dot": result["theta_dot"][:, 0]}


if __name__ == "__main__":
    print("=== Validation 1: FFT peak-frequency detection accuracy ===")
    dt = 0.001
    duration = 5.0
    n = int(duration / dt)
    t = np.arange(n) * dt
    freq_resolution = 1.0 / (n * dt)
    print(f"Sampling: dt={dt}s, N={n}, frequency resolution = {freq_resolution:.4f} Hz")

    test_freqs = [5.0, 12.37, 23.0, 47.5]  # includes a non-bin-aligned frequency
    max_freq_error = 0.0
    for f_true in test_freqs:
        signal = np.sin(2 * np.pi * f_true * t)
        freqs, mag = compute_spectrum(signal, dt)
        f_detected = find_peak_frequency(freqs, mag, f_min=0.5)
        err = abs(f_detected - f_true)
        max_freq_error = max(max_freq_error, err)
        print(f"  true={f_true:6.2f} Hz  detected={f_detected:6.2f} Hz  error={err:.4f} Hz")

    assert max_freq_error <= freq_resolution, (
        f"Peak detection error ({max_freq_error:.4f} Hz) exceeds frequency "
        f"resolution ({freq_resolution:.4f} Hz)."
    )
    print(f"All peaks detected within frequency resolution ({freq_resolution:.4f} Hz).")

    print("\n=== Validation 2: amplitude accuracy check ===")
    A_true = 2.5
    f_true = 10.0
    signal = A_true * np.sin(2 * np.pi * f_true * t)
    freqs, mag = compute_spectrum(signal, dt)
    peak_idx = np.argmax(mag)
    A_detected = mag[peak_idx]
    amp_err = abs(A_detected - A_true) / A_true
    print(f"  true amplitude={A_true}  detected amplitude={A_detected:.4f}  "
          f"relative error={amp_err*100:.2f}%")
    assert amp_err < 0.05, "Amplitude estimate should be accurate to within 5% with Hann correction."
    print("Amplitude estimate verified accurate (Hann window correction working).")

    print("\n=== Validation 3: healthy joint -- no false positive ===")
    healthy_ref = simulate_vibration_signal(drive_freq_hz=8.0, duration=duration, dt=dt)
    healthy_test = simulate_vibration_signal(drive_freq_hz=8.0, duration=duration, dt=dt)
    result_healthy = detect_spectral_anomalies(
        healthy_ref["theta_dot"], healthy_test["theta_dot"], dt
    )
    print(f"  Healthy vs healthy -- is_anomalous: {result_healthy['is_anomalous']} (expected False)")
    print(f"  Number of flagged bins: {result_healthy['anomaly_mask'].sum()}")
    assert not result_healthy["is_anomalous"], "Two healthy signals should not trigger a false alarm."
    print("No false positive on healthy-vs-healthy comparison.")

    print("\n=== Validation 4: fault injection -- correctly detected ===")
    fault_freq = 37.0  # a fault frequency unrelated to the 8 Hz drive frequency
    faulty_signal = simulate_vibration_signal(
        drive_freq_hz=8.0, duration=duration, dt=dt,
        fault_freq_hz=fault_freq, fault_amplitude=0.03,
    )
    result_fault = detect_spectral_anomalies(
        healthy_ref["theta_dot"], faulty_signal["theta_dot"], dt
    )
    print(f"  Healthy vs faulty -- is_anomalous: {result_fault['is_anomalous']} (expected True)")
    print(f"  Flagged frequencies: {np.round(result_fault['anomaly_freqs'], 2)}")
    assert result_fault["is_anomalous"], "Injected fault frequency should be detected."

    detected_near_fault = np.any(np.abs(result_fault["anomaly_freqs"] - fault_freq) < 1.0)
    print(f"  Detected frequency near true fault freq ({fault_freq} Hz): {detected_near_fault}")
    assert detected_near_fault, "The flagged frequency should be near the true injected fault frequency."
    print("Fault correctly detected at the right frequency.")

    print("\n=== Validation 5: pure noise -- no false positive ===")
    rng = np.random.default_rng(0)
    noise_ref = rng.normal(0, 0.01, n)
    noise_test = rng.normal(0, 0.01, n)
    result_noise = detect_spectral_anomalies(noise_ref, noise_test, dt)
    print(f"  Noise vs noise -- is_anomalous: {result_noise['is_anomalous']} (expected False)")
    print(f"  Number of flagged bins: {result_noise['anomaly_mask'].sum()} / {len(result_noise['freqs'])}")
    assert not result_noise["is_anomalous"], "Random noise should not trigger a false alarm."
    print("No false positive on pure-noise comparison.")

    print("\nAll checks passed.")