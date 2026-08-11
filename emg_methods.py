from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """Remove the mean value from a 1D signal."""
    return x - np.mean(x)


def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1D signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, x)


def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a notch filter for mains interference."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.sosfiltfilt(signal.tf2sos(b, a), x)


def full_wave_rectification(x: np.ndarray) -> np.ndarray:
    """Convert a signal to its absolute value."""
    return np.abs(x)


def moving_average_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a moving-average envelope."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="same")


def rms_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a sliding-window RMS envelope."""
    pad = window_size // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    rms = np.empty_like(x)
    for i in range(len(x)):
        window = padded[i : i + window_size]
        rms[i] = np.sqrt(np.mean(window**2))
    return rms


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute a Hilbert-transform based amplitude envelope."""
    analytic_signal = signal.hilbert(x)
    return np.abs(analytic_signal)


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


def compute_time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Compute a compact set of time-domain EMG features."""
    return {
        "MAV": mean_absolute_value(x),
        "RMS": root_mean_square(x),
        "Variance": variance_value(x),
        "Std": std_value(x),
        "IEMG": iemg(x),
        "WL": waveform_length(x),
        "ZC": float(zero_crossings(x)),
        "SSC": float(slope_sign_changes(x)),
        "WAMP": float(willison_amplitude(x)),
        "Peak Amplitude": peak_amplitude(x),
        "Signal Energy": signal_energy(x),
    }


def compute_fft_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided FFT magnitude spectrum."""
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude


def compute_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Welch PSD estimate."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
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


def compute_frequency_domain_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute frequency-domain EMG descriptors."""
    freqs, psd = compute_psd(x, fs)
    return {
        "MNF": mean_frequency(freqs, psd),
        "MDF": median_frequency(freqs, psd),
        "Peak Frequency": peak_frequency(freqs, psd),
        "Total Spectral Power": total_spectral_power(psd),
    }
