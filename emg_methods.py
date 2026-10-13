from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def remove_dc_offset(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove the mean value from a signal (1D, or along `axis` for multichannel data)."""
    return x - np.mean(x, axis=axis, keepdims=True)


def default_emg_band(fs: float, lowcut: float = 20.0, highcut: float = 450.0) -> Tuple[float, float]:
    """Return a band-pass range that is valid for the given sampling frequency.

    The textbook 20-450 Hz surface EMG band needs fs >= ~1 kHz. Low-cost wearables
    (Myo armband: 200 Hz, BrainAccess: 250 Hz) cannot represent frequencies above
    fs/2, so the upper edge is clipped to 90 % of the Nyquist frequency.
    """
    nyquist = 0.5 * fs
    return lowcut, min(highcut, 0.9 * nyquist)


def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            f"Invalid band {lowcut}-{highcut} Hz for fs={fs} Hz (Nyquist {nyquist} Hz). "
            "Use default_emg_band(fs) to get a valid range."
        )
    sos = signal.butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def notch_filter(
    x: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0, axis: int = -1
) -> np.ndarray:
    """Apply a notch filter for mains interference."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.sosfiltfilt(signal.tf2sos(b, a), x, axis=axis)


def mains_notch_filter(
    x: np.ndarray, fs: float, mains: float = 50.0, quality: float = 30.0, axis: int = -1
) -> np.ndarray:
    """Notch the mains frequency and every harmonic below the Nyquist frequency."""
    nyquist = 0.5 * fs
    harmonic = mains
    while harmonic < 0.98 * nyquist:
        x = notch_filter(x, fs, freq=harmonic, quality=quality, axis=axis)
        harmonic += mains
    return x


def preprocess_emg(
    x: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
    mains: Optional[float] = 50.0,
    axis: int = -1,
) -> Dict[str, np.ndarray]:
    """Run the standard pipeline: DC removal -> band-pass -> mains notch -> rectification.

    Returns every intermediate stage so notebooks can plot them side by side.
    """
    band = band or default_emg_band(fs)
    x_dc = remove_dc_offset(x, axis=axis)
    x_bp = butter_bandpass_filter(x_dc, fs, *band, axis=axis)
    x_notch = mains_notch_filter(x_bp, fs, mains=mains, axis=axis) if mains else x_bp
    return {
        "dc_removed": x_dc,
        "bandpassed": x_bp,
        "filtered": x_notch,
        "rectified": full_wave_rectification(x_notch),
    }


def full_wave_rectification(x: np.ndarray) -> np.ndarray:
    """Convert a signal to its absolute value."""
    return np.abs(x)


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


def moving_average_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a moving-average envelope."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="same")


def rms_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a sliding-window RMS envelope."""
    pad = window_size // 2
    padded = np.pad(x, (pad, window_size - 1 - pad), mode="reflect")
    windows = sliding_window_view(padded**2, window_size)
    return np.sqrt(windows.mean(axis=-1))


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute a Hilbert-transform based amplitude envelope."""
    analytic_signal = signal.hilbert(x)
    return np.abs(analytic_signal)


def lowpass_envelope(x: np.ndarray, fs: float, cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """Linear envelope: low-pass filter of the rectified signal (common in biomechanics)."""
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.abs(x))


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------


def mean_absolute_value(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def root_mean_square(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def variance_value(x: np.ndarray) -> float:
    return float(np.var(x))


def std_value(x: np.ndarray) -> float:
    return float(np.std(x))


def iemg(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)))


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def zero_crossings(x: np.ndarray, threshold: float = 0.0) -> int:
    signs = np.sign(x - threshold)
    return int(np.sum(signs[:-1] != signs[1:]))


def slope_sign_changes(x: np.ndarray, threshold: float = 0.0) -> int:
    dx = np.diff(x)
    signs = np.sign(dx)
    return int(np.sum((signs[:-1] != signs[1:]) & (np.abs(dx[:-1]) > threshold)))


def willison_amplitude(x: np.ndarray, threshold: float = 0.01) -> int:
    return int(np.sum(np.abs(np.diff(x)) > threshold))


def peak_amplitude(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def signal_energy(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def compute_time_domain_features(x: np.ndarray, wamp_threshold: float = 0.01) -> Dict[str, float]:
    """Compute a compact set of time-domain EMG features.

    Use the band-pass filtered (not rectified) signal: ZC and SSC are meaningless
    on a rectified signal. `wamp_threshold` is in the same unit as `x`.
    """
    return {
        "MAV": mean_absolute_value(x),
        "RMS": root_mean_square(x),
        "Variance": variance_value(x),
        "Std": std_value(x),
        "IEMG": iemg(x),
        "WL": waveform_length(x),
        "ZC": float(zero_crossings(x)),
        "SSC": float(slope_sign_changes(x)),
        "WAMP": float(willison_amplitude(x, wamp_threshold)),
        "Peak Amplitude": peak_amplitude(x),
        "Signal Energy": signal_energy(x),
    }


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------


def compute_fft_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided FFT magnitude spectrum."""
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude


def compute_psd(x: np.ndarray, fs: float, nperseg: int = 1024, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Welch PSD estimate."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(nperseg, x.shape[axis]), axis=axis)
    return freqs, psd


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    return float(np.sum(freqs * psd) / np.sum(psd))


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    if total_power == 0:
        return 0.0
    threshold = 0.5 * total_power
    idx = np.where(cumulative_power >= threshold)[0][0]
    return float(freqs[idx])


def peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    return float(freqs[np.argmax(psd)])


def total_spectral_power(psd: np.ndarray) -> float:
    return float(np.sum(psd))


def compute_frequency_domain_features(x: np.ndarray, fs: float, nperseg: int = 1024) -> Dict[str, float]:
    """Compute frequency-domain EMG descriptors."""
    freqs, psd = compute_psd(x, fs, nperseg=nperseg)
    return {
        "MNF": mean_frequency(freqs, psd),
        "MDF": median_frequency(freqs, psd),
        "Peak Frequency": peak_frequency(freqs, psd),
        "Total Spectral Power": total_spectral_power(psd),
    }


# ---------------------------------------------------------------------------
# Windowed (vectorised) features for machine learning
# ---------------------------------------------------------------------------


def window_starts(n_samples: int, window_size: int, step: int) -> np.ndarray:
    """Start indices of all complete windows."""
    if n_samples < window_size:
        return np.empty(0, dtype=int)
    return np.arange(0, n_samples - window_size + 1, step)


def windowed_features(
    x: np.ndarray,
    fs: float,
    window_size: int,
    step: int,
    channel_names: Optional[Sequence[str]] = None,
    wamp_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compute classic EMG features over sliding windows for every channel.

    Parameters
    ----------
    x : array of shape (n_channels, n_samples) or (n_samples,)
        Band-pass filtered (not rectified) EMG.
    wamp_threshold : float, optional
        WAMP/SSC threshold in signal units. Defaults to 10 % of each channel's
        overall standard deviation, which adapts to the device's amplitude scale.

    Returns
    -------
    DataFrame with one row per window and columns `<channel>_<feature>`,
    plus `start` (sample index) and `t_center` (seconds).
    """
    x = np.atleast_2d(x)
    n_ch, n = x.shape
    channel_names = list(channel_names) if channel_names is not None else [f"ch{i + 1}" for i in range(n_ch)]
    starts = window_starts(n, window_size, step)
    win = sliding_window_view(x, window_size, axis=-1)[:, starts, :]  # (ch, n_win, win)

    thr = np.std(x, axis=-1)[:, None] * 0.1 if wamp_threshold is None else np.full((n_ch, 1), wamp_threshold)
    dx = np.diff(win, axis=-1)
    signs = np.sign(win)
    dsigns = np.sign(dx)

    feats = {
        "MAV": np.mean(np.abs(win), axis=-1),
        "RMS": np.sqrt(np.mean(win**2, axis=-1)),
        "VAR": np.var(win, axis=-1),
        "WL": np.sum(np.abs(dx), axis=-1),
        "ZC": np.sum(signs[..., :-1] != signs[..., 1:], axis=-1).astype(float),
        "SSC": np.sum((dsigns[..., :-1] != dsigns[..., 1:]) & (np.abs(dx[..., :-1]) > thr[..., None]), axis=-1).astype(float),
        "WAMP": np.sum(np.abs(dx) > thr[..., None], axis=-1).astype(float),
    }

    # Spectral features from a Hann-windowed periodogram of each window.
    tapered = (win - win.mean(axis=-1, keepdims=True)) * np.hanning(window_size)
    psd = np.abs(np.fft.rfft(tapered, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
    total = psd.sum(axis=-1)
    safe_total = np.where(total > 0, total, 1.0)
    feats["MNF"] = (psd * freqs).sum(axis=-1) / safe_total
    cumulative = np.cumsum(psd, axis=-1) / safe_total[..., None]
    feats["MDF"] = freqs[np.argmax(cumulative >= 0.5, axis=-1)]

    columns = {}
    for ci, name in enumerate(channel_names):
        for fname, values in feats.items():
            columns[f"{name}_{fname}"] = values[ci]
    out = pd.DataFrame(columns)
    out.insert(0, "start", starts)
    out.insert(1, "t_center", (starts + window_size / 2) / fs)
    return out


# ---------------------------------------------------------------------------
# Activation detection and fatigue analysis
# ---------------------------------------------------------------------------


def detect_activations(
    envelope: np.ndarray,
    fs: float,
    k: float = 3.0,
    baseline: Optional[Tuple[int, int]] = None,
    min_duration_s: float = 0.2,
    min_gap_s: float = 0.2,
    baseline_s: float = 1.0,
) -> pd.DataFrame:
    """Threshold-based onset/offset detection (baseline mean + k * SD).

    If `baseline` (start, stop sample indices) is not given, the envelope is
    split into non-overlapping `baseline_s` windows and the quietest 20 % of
    them are pooled as the resting baseline. Pooling several rest periods makes
    the SD reflect rest-to-rest variation, not just one quiet second. Bursts closer than
    `min_gap_s` are merged and bursts shorter than `min_duration_s` are dropped.
    Returns a DataFrame with onset/offset sample indices and times.
    """
    if baseline is not None:
        ref = envelope[baseline[0] : baseline[1]]
    else:
        win = max(1, min(int(baseline_s * fs), len(envelope)))
        blocks = envelope[: len(envelope) // win * win].reshape(-1, win)
        quiet = np.argsort(blocks.mean(axis=1))[: max(1, int(np.ceil(0.2 * len(blocks))))]
        ref = blocks[quiet].ravel()
    threshold = float(np.mean(ref) + k * np.std(ref))

    active = envelope > threshold
    edges = np.diff(active.astype(int), prepend=0, append=0)
    onsets = np.flatnonzero(edges == 1)
    offsets = np.flatnonzero(edges == -1)

    merged = []
    for on, off in zip(onsets, offsets):
        if merged and on - merged[-1][1] < min_gap_s * fs:
            merged[-1][1] = off
        else:
            merged.append([on, off])
    bursts = [(on, off) for on, off in merged if off - on >= min_duration_s * fs]

    df = pd.DataFrame(bursts, columns=["onset", "offset"], dtype=int)
    df["onset_s"] = df["onset"] / fs
    df["offset_s"] = df["offset"] / fs
    df["duration_s"] = df["offset_s"] - df["onset_s"]
    df.attrs["threshold"] = threshold
    return df


def activity_mask(n_samples: int, bursts: pd.DataFrame) -> np.ndarray:
    """Boolean mask that is True inside detected bursts."""
    mask = np.zeros(n_samples, dtype=bool)
    for on, off in zip(bursts["onset"], bursts["offset"]):
        mask[on:off] = True
    return mask


def spectral_fatigue_track(
    x: np.ndarray, fs: float, window_s: float = 1.0, step_s: float = 0.5, nperseg: int = 256
) -> pd.DataFrame:
    """Track MNF and MDF over time; a downward trend during sustained contraction indicates fatigue."""
    window = int(window_s * fs)
    step = int(step_s * fs)
    rows = []
    for start in window_starts(len(x), window, step):
        seg = x[start : start + window]
        freqs, psd = compute_psd(seg, fs, nperseg=nperseg)
        rows.append(
            {
                "t_s": (start + window / 2) / fs,
                "RMS": root_mean_square(seg),
                "MNF": mean_frequency(freqs, psd),
                "MDF": median_frequency(freqs, psd),
            }
        )
    return pd.DataFrame(rows)
