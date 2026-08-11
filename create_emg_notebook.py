import json
from pathlib import Path
import textwrap

workspace = Path('/home/michalkubascik/Research/Litva/EMG_signal_processing')
out_dir = workspace / 'notebooks'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'emg_signal_processing_tutorial.ipynb'

cells = []


def add_markdown(text: str):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': textwrap.dedent(text).strip().splitlines(keepends=True),
    })


def add_code(code: str):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': textwrap.dedent(code).strip().splitlines(keepends=True),
    })


add_markdown(r'''
# EMG Signal Processing Tutorial for Biomedical Signal Processing

This notebook is a complete, teaching-oriented workflow for processing Electromyography (EMG) signals in Python. It moves from raw signal acquisition to preprocessing, envelope extraction, feature engineering, visualization, and interpretation.

The notebook is designed for students who want to understand both the physiology behind EMG and the computational steps needed to analyze it.
''')

add_markdown(r'''
## Learning Objectives

By the end of this notebook, you will be able to:

- understand the basic physiology of EMG signals;
- load and explore raw EMG data;
- apply common preprocessing steps (DC removal, filtering, rectification);
- extract envelopes and time-domain / frequency-domain features;
- interpret the biological meaning of different EMG descriptors;
- use EMG features in a simple machine-learning workflow.
''')

add_markdown(r'''
## Dataset Selection

A good educational EMG dataset should contain raw or minimally processed signals, have a known sampling frequency, and be easy to load with Python. Public Kaggle datasets are useful because they are broadly accessible and often include realistic noise and motion artifacts.

For this notebook, the code will try to download a public Kaggle EMG dataset automatically. If Kaggle authentication is not available or the dataset is temporarily unavailable, the notebook falls back to a reproducible synthetic EMG-like signal so the entire workflow can still run end-to-end.

### Why this is suitable for teaching

- it exposes students to realistic signal morphology;
- it includes amplitude variations that resemble muscle activation bursts;
- it supports both time-domain and frequency-domain analysis;
- it is a good example of how preprocessing can change interpretation.

### Kaggle API authentication

If you want to download a real public Kaggle dataset, create a Kaggle API token first:

1. Sign in to Kaggle.
2. Open your account settings.
3. Download `kaggle.json`.
4. Place it in `~/.kaggle/kaggle.json`.

The notebook below will check for that file and download the dataset automatically if it exists.
''')

add_code(r'''
# Environment setup and package installation
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Optional packages for machine learning extension
for package in ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn', 'kaggle']:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Re-import after installation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120
print('Environment ready.')
''')

add_code(r'''
# Download a public Kaggle dataset if possible, otherwise fall back to a synthetic EMG signal.
from pathlib import Path
import os
import shutil
import subprocess
import zipfile

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# Candidate Kaggle dataset slugs. The code tries each one until one downloads successfully.
DATASET_SLUGS = [
    'sukrit/surface-emg-dataset',
    'fakher/semg-surface-electromyography',
    'mohamed/emg-dataset',
]


def download_kaggle_dataset(dataset_slug: str, out_dir: Path) -> Optional[Path]:
    """Download a Kaggle dataset into out_dir and return the extracted folder path."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print('Kaggle API credentials were not found. Falling back to a synthetic dataset.')
        return None

    os.environ['KAGGLE_CONFIG_DIR'] = str(Path.home() / '.kaggle')
    download_dir = out_dir / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', str(download_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f'Failed to download {dataset_slug}: {exc.stderr.strip()}')
        return None

    zip_files = list(download_dir.glob('*.zip'))
    if not zip_files:
        return None

    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(download_dir)
        zip_path.unlink(missing_ok=True)

    # Return the first non-empty directory inside the downloaded folder.
    for candidate in sorted(download_dir.iterdir()):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return download_dir


def build_synthetic_emg_signal() -> pd.DataFrame:
    """Create a synthetic, EMG-like signal with realistic bursts and noise."""
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    base = 0.15 * np.sin(2 * np.pi * 0.3 * t)
    bursts = np.zeros_like(t)
    burst_times = [0.8, 2.2, 3.4, 5.0, 6.9, 8.4]
    for bt in burst_times:
        idx = np.where((t >= bt) & (t < bt + 0.6))[0]
        bursts[idx] += np.sin(np.pi * np.linspace(0, 1, len(idx)))
    noise = 0.12 * np.random.randn(len(t))
    emg = base + bursts + noise

    df = pd.DataFrame({
        'time_s': t,
        'emg_uV': emg,
        'activity_label': (np.abs(emg) > 0.25).astype(int),
    })
    return df


def load_emg_dataframe() -> tuple[pd.DataFrame, float]:
    """Load EMG data from Kaggle if available; otherwise use the synthetic fallback."""
    for slug in DATASET_SLUGS:
        extracted_dir = download_kaggle_dataset(slug, DATA_DIR)
        if extracted_dir is None:
            continue

        candidates = []
        for pattern in ['*.csv', '*.txt', '*.tsv', '*.parquet', '*.mat']:
            candidates.extend(extracted_dir.rglob(pattern))
        if not candidates:
            continue

        # Prefer CSV files and select a likely EMG column.
        for path in candidates:
            if path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if df.empty:
                    continue
                # Try to identify EMG-related columns.
                col_candidates = [c for c in df.columns if 'emg' in c.lower() or 'signal' in c.lower() or 'data' in c.lower()]
                if not col_candidates:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col_candidates = [numeric_cols[0]]
                if col_candidates:
                    emg_col = col_candidates[0]
                    time_col = next((c for c in df.columns if 'time' in c.lower() or 'sample' in c.lower()), None)
                    if time_col is None:
                        fs = 1000.0
                        t = np.arange(len(df)) / fs
                        out_df = pd.DataFrame({'time_s': t, 'emg_uV': df[emg_col].astype(float)})
                    else:
                        out_df = pd.DataFrame({'time_s': df[time_col].astype(float), 'emg_uV': df[emg_col].astype(float)})
                    return out_df, float(np.median(np.diff(out_df['time_s'].dropna().to_numpy())))

        print(f'Loaded data from {extracted_dir} but the format was not recognized; using synthetic fallback instead.')
        break

    print('Using the synthetic EMG fallback dataset for reproducibility.')
    return build_synthetic_emg_signal(), 1000.0


df, inferred_dt = load_emg_dataframe()

# If the dataframe is too short, pad it with synthetic data.
if len(df) < 200:
    extra = build_synthetic_emg_signal().iloc[:200-len(df)]
    df = pd.concat([df, extra], ignore_index=True)

# Ensure a clean time axis if missing.
if 'time_s' not in df.columns:
    df['time_s'] = np.arange(len(df)) / 1000.0

# Sampling frequency estimate from time vector.
if 'time_s' in df.columns and df['time_s'].nunique() > 1:
    dt = np.median(np.diff(np.sort(df['time_s'].to_numpy())))
    sampling_frequency = 1 / dt if dt > 0 else 1000.0
else:
    sampling_frequency = 1000.0

# Use a standard EMG amplitude column.
if 'emg_uV' not in df.columns:
    emg_col = [c for c in df.columns if 'emg' in c.lower() or 'signal' in c.lower() or 'data' in c.lower()]
    if emg_col:
        df['emg_uV'] = df[emg_col[0]].astype(float)
    else:
        df['emg_uV'] = pd.to_numeric(df.iloc[:, 0], errors='coerce').astype(float)

signal_raw = df['emg_uV'].to_numpy(dtype=float)
print(f'Loaded {len(df)} samples.')
print(f'Estimated sampling frequency: {sampling_frequency:.1f} Hz')
df.head()
''')

add_markdown(r'''
## Data Exploration

Before processing, it is important to understand the shape of the data and the basic properties of the signal.

We will inspect:

- the dataset shape and columns;
- the number of missing values;
- the signal length and sampling frequency;
- descriptive statistics of the EMG amplitude.
''')

add_code(r'''
# Dataset overview and basic statistics.
print('Dataset shape:', df.shape)
print('\nColumn names:')
print(df.columns.tolist())
print('\nMissing values:')
print(df.isna().sum())
print('\nSummary statistics:')
print(df.describe().round(4))

print(f'\nSignal length: {len(signal_raw)} samples')
print(f'Sampling frequency: {sampling_frequency:.1f} Hz')
print(f'Time span: {df["time_s"].iloc[-1] - df["time_s"].iloc[0]:.3f} s')
''')

add_code(r'''
# Visualize the raw EMG signal in three ways: full overview, zoomed segment, and amplitude histogram.
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

axes[0].plot(df['time_s'], df['emg_uV'], color='tab:blue', linewidth=1.0)
axes[0].set_title('Raw EMG Signal - Full Overview')
axes[0].set_ylabel('Amplitude')
axes[0].set_xlabel('Time (s)')
axes[0].grid(True, alpha=0.3)

# Zoomed segment
start_idx = max(0, len(df)//4)
end_idx = min(len(df), start_idx + int(0.5 * sampling_frequency))
axes[1].plot(df['time_s'].iloc[start_idx:end_idx], df['emg_uV'].iloc[start_idx:end_idx], color='tab:orange')
axes[1].set_title('Raw EMG Signal - Zoomed Segment')
axes[1].set_ylabel('Amplitude')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)

# Histogram
axes[2].hist(df['emg_uV'], bins=60, color='tab:green', edgecolor='black', alpha=0.8)
axes[2].set_title('Histogram of Raw EMG Amplitudes')
axes[2].set_xlabel('Amplitude')
axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
''')

add_markdown(r'''
## What to Observe in the Plots

- The full overview shows the general morphology of the EMG signal, including bursts of activity and periods of lower amplitude.
- The zoomed segment highlights local structure, such as transient bursts and noise.
- The histogram shows the distribution of amplitudes and helps reveal whether the signal has a strong baseline offset or heavy-tailed noise.
''')

add_markdown(r'''
## EMG Preprocessing

Raw EMG signals are often contaminated by artifacts, drift, and power-line interference. Preprocessing is therefore essential before any meaningful interpretation.

### Step 1: Remove DC offset
The mean of the signal is subtracted so the baseline is centered around zero. This is important because a large offset can dominate amplitude-based features.

### Step 2: Band-pass filtering
A Butterworth band-pass filter in the range 20-450 Hz is commonly used for EMG, because this range removes low-frequency drift and high-frequency noise while preserving the physiologically relevant band.

### Step 3: Notch filtering
Power-line interference (commonly 50 Hz or 60 Hz) is removed to avoid strong line-frequency contamination.

### Step 4: Full-wave rectification
The signal is converted to absolute values. Rectification helps make positive-going activation bursts more interpretable in amplitude analysis.
''')

add_code(r'''
# Reusable preprocessing functions

def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """Remove the mean value from a signal."""
    return x - np.mean(x)


def butter_bandpass_filter(x: np.ndarray, fs: float, lowcut: float = 20.0, highcut: float = 450.0,
                           order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1D signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, x)


def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a notch filter at the specified mains frequency."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.sosfiltfilt(signal.tf2sos(b, a), x)


def full_wave_rectification(x: np.ndarray) -> np.ndarray:
    """Convert a signal to its absolute value."""
    return np.abs(x)


# Apply the preprocessing steps
x_dc = remove_dc_offset(signal_raw)
x_bp = butter_bandpass_filter(x_dc, sampling_frequency)
x_notch = notch_filter(x_bp, sampling_frequency, freq=50.0)
x_rectified = full_wave_rectification(x_notch)

# Create a comparison figure
fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

axes[0].plot(df['time_s'], signal_raw, color='tab:blue', label='Raw', linewidth=1.0)
axes[0].set_title('EMG Preprocessing Stages')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right')

axes[1].plot(df['time_s'], x_dc, color='tab:orange', label='DC-corrected', linewidth=1.0)
axes[1].set_ylabel('Amplitude')
axes[1].legend(loc='upper right')

axes[2].plot(df['time_s'], x_bp, color='tab:green', label='Band-pass filtered', linewidth=1.0)
axes[2].set_ylabel('Amplitude')
axes[2].legend(loc='upper right')

axes[3].plot(df['time_s'], x_notch, color='tab:red', label='Notch-filtered', linewidth=1.0)
axes[3].set_ylabel('Amplitude')
axes[3].legend(loc='upper right')

axes[4].plot(df['time_s'], x_rectified, color='tab:purple', label='Rectified', linewidth=1.0)
axes[4].set_ylabel('Amplitude')
axes[4].set_xlabel('Time (s)')
axes[4].legend(loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
''')

add_markdown(r'''
## Envelope Detection

After preprocessing, an envelope is often used to summarize the amplitude evolution of the EMG signal over time. The envelope highlights bursts of muscle activity and is widely used in prosthetics, rehabilitation, and biomechanics.

### Moving average envelope
A moving average smooths the rectified signal over a short window. This is simple and interpretable, but it is sensitive to window choice.

### RMS envelope
The RMS envelope is computed over sliding windows and is especially useful because it reflects power-like amplitude changes.

### Hilbert envelope
The Hilbert transform provides an analytic signal whose magnitude gives an amplitude envelope. It is theoretically elegant and often produces a smooth envelope.
''')

add_code(r'''
# Envelope extraction functions

def moving_average_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a moving average envelope."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode='same')


def rms_envelope(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Compute a sliding-window RMS envelope."""
    pad = window_size // 2
    padded = np.pad(x, (pad, pad), mode='reflect')
    rms = np.empty_like(x)
    for i in range(len(x)):
        window = padded[i:i + window_size]
        rms[i] = np.sqrt(np.mean(window**2))
    return rms


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute the Hilbert transform magnitude envelope."""
    analytic_signal = signal.hilbert(x)
    return np.abs(analytic_signal)


ma_env = moving_average_envelope(x_rectified, window_size=80)
rms_env = rms_envelope(x_rectified, window_size=80)
hilbert_env = hilbert_envelope(x_rectified)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['time_s'], x_rectified, label='Rectified signal', color='gray', linewidth=0.8)
ax.plot(df['time_s'], ma_env, label='Moving average envelope', color='tab:blue', linewidth=1.5)
ax.plot(df['time_s'], rms_env, label='RMS envelope', color='tab:orange', linewidth=1.5)
ax.plot(df['time_s'], hilbert_env, label='Hilbert envelope', color='tab:green', linewidth=1.5)
ax.set_title('Comparison of EMG Envelope Extraction Methods')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
''')

add_markdown(r'''
## Time-Domain Feature Extraction

Time-domain features are simple and fast to compute. They are among the most common descriptors used in EMG-based classification and prosthetic control.

### Common features

- Mean Absolute Value (MAV): reflects average activation intensity.
- Root Mean Square (RMS): sensitive to signal power and amplitude.
- Variance and Standard Deviation: indicate dispersion.
- Integrated EMG (IEMG): captures cumulative signal magnitude.
- Waveform Length (WL): measures signal complexity.
- Zero Crossings (ZC): count sign changes, often related to frequency content.
- Slope Sign Changes (SSC): capture changes in slope.
- Willison Amplitude (WAMP): counts threshold crossings.
- Peak Amplitude: reflects maximum activation.
- Signal Energy: captures total power-like content.
''')

add_code(r'''
# Time-domain feature functions

def mean_absolute_value(x: np.ndarray) -> float:
    return np.mean(np.abs(x))


def root_mean_square(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))


def variance_value(x: np.ndarray) -> float:
    return np.var(x)


def std_value(x: np.ndarray) -> float:
    return np.std(x)


def iemg(x: np.ndarray) -> float:
    return np.sum(np.abs(x))


def waveform_length(x: np.ndarray) -> float:
    return np.sum(np.abs(np.diff(x)))


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
    return np.max(np.abs(x))


def signal_energy(x: np.ndarray) -> float:
    return np.sum(x**2)


# Compute time-domain features on the rectified signal.
feature_values = {
    'MAV': mean_absolute_value(x_rectified),
    'RMS': root_mean_square(x_rectified),
    'Variance': variance_value(x_rectified),
    'Std': std_value(x_rectified),
    'IEMG': iemg(x_rectified),
    'WL': waveform_length(x_rectified),
    'ZC': zero_crossings(x_rectified),
    'SSC': slope_sign_changes(x_rectified),
    'WAMP': willison_amplitude(x_rectified),
    'Peak Amplitude': peak_amplitude(x_rectified),
    'Signal Energy': signal_energy(x_rectified),
}

feature_values
''')

add_markdown(r'''
## Frequency-Domain Features

Frequency-domain features help us characterize how energy is distributed across frequencies. They are especially useful for fatigue analysis and for understanding how signal content changes during sustained muscle activity.

### FFT and PSD
The Fast Fourier Transform (FFT) is used to inspect the spectrum of the signal. The Power Spectral Density (PSD) summarizes how power is distributed over frequency.

### Mean Frequency and Median Frequency
These summarize the center of the spectral distribution. They are often used in fatigue studies because the spectrum shifts when muscles become tired.
''')

add_code(r'''
# Frequency-domain feature functions

def compute_fft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided FFT magnitude spectrum."""
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude


def compute_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PSD using Welch's method."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    return freqs, psd


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    return np.sum(freqs * psd) / np.sum(psd)


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    if total_power == 0:
        return 0.0
    threshold = 0.5 * total_power
    idx = np.where(cumulative_power >= threshold)[0][0]
    return freqs[idx]


def peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    return freqs[np.argmax(psd)]


def total_spectral_power(psd: np.ndarray) -> float:
    return np.sum(psd)


freqs, magnitude = compute_fft(x_bp, sampling_frequency)
psd_freqs, psd_vals = compute_psd(x_bp, sampling_frequency)

freq_features = {
    'MNF': mean_frequency(psd_freqs, psd_vals),
    'MDF': median_frequency(psd_freqs, psd_vals),
    'Peak Frequency': peak_frequency(psd_freqs, psd_vals),
    'Total Spectral Power': total_spectral_power(psd_vals),
}

# Plot the spectrum and PSD
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(freqs, magnitude, color='tab:blue')
axes[0].set_title('EMG Frequency Spectrum')
axes[0].set_ylabel('Magnitude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(psd_freqs, psd_vals, color='tab:orange')
axes[1].set_title('Power Spectral Density')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

freq_features
''')

add_markdown(r'''
## Feature Summary

The extracted features can be summarized in a DataFrame so the student can compare them easily. A concise table helps connect each value to a biological interpretation.
''')

add_code(r'''
# Build a feature summary table.
feature_summary = pd.DataFrame({
    'feature': list(feature_values.keys()) + list(freq_features.keys()),
    'value': list(feature_values.values()) + list(freq_features.values()),
})
feature_summary['interpretation'] = [
    'Measures average amplitude intensity',
    'Reflects signal power and amplitude',
    'Measures dispersion around the mean',
    'Quantifies variability',
    'Summarizes cumulative magnitude',
    'Captures waveform complexity',
    'Counts sign changes, related to frequency content',
    'Measures slope changes',
    'Counts threshold-crossing events',
    'Shows maximum activation amplitude',
    'Reflects total signal energy',
    'Center of spectral distribution',
    'Median spectral frequency',
    'Dominant spectral peak',
    'Total power in the spectrum',
]

feature_summary.round(4)
''')

add_markdown(r'''
## Interpretation of Results

### What each feature tells us

- Amplitude-based features such as MAV, RMS, and peak amplitude reflect the intensity of muscle activation.
- Variance and standard deviation indicate how much the signal fluctuates over time.
- IEMG and signal energy are related to the total amount of electrical activity.
- Waveform length and slope sign changes describe the complexity and dynamic structure of the signal.
- Frequency-domain features reveal how energy is distributed across frequency, which can be informative for fatigue analysis.

### Fatigue-related interpretation

During sustained contractions, muscle fatigue often shifts the EMG spectrum to lower frequencies and reduces median frequency. This is why MDF and MNF are often used in fatigue studies.

### Classification applications

In gesture recognition and prosthesis control, features such as MAV, RMS, WL, ZC, SSC, and WAMP are commonly used as inputs to classifiers.

### Limitations

- Time-domain features are easy to compute but can be influenced by noise and electrode placement.
- Frequency-domain features require a sufficient signal duration and careful preprocessing.
- Features can be sensitive to the chosen window length and the recording conditions.
''')

add_markdown(r'''
## Practical Applications

EMG signal processing has many real-world applications:

- Gesture recognition
- Prosthetic control
- Human-computer interaction
- Rehabilitation engineering
- Sports biomechanics
- Muscle fatigue monitoring
- Medical diagnostics

In each application, preprocessing and feature extraction are critical for obtaining reliable and interpretable outcomes.
''')

add_markdown(r'''
## Bonus Features

Additional descriptors can be useful for more advanced analysis:

- Log Detector: a simple amplitude-based feature that emphasizes moderate activation.
- Hjorth Parameters: mobility and complexity descriptors derived from the signal's variance and derivatives.
- Entropy measures: quantify signal irregularity.
- Spectral entropy: characterizes the spread of spectral power.
- Teager-Kaiser Energy Operator: useful for detecting transient bursts.
- Autoregressive coefficients: model the signal as an autoregressive process.

These features are often more advanced, but they can be valuable once students are comfortable with the basic descriptors above.
''')

add_code(r'''
# Optional bonus features for more advanced study

def log_detector(x: np.ndarray) -> float:
    return np.mean(np.log1p(np.abs(x)))


def hjorth_parameters(x: np.ndarray) -> tuple[float, float]:
    first_diff = np.diff(x)
    var_x = np.var(x)
    var_dx = np.var(first_diff)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    complexity = np.sqrt(np.var(np.diff(first_diff)) / var_dx) if var_dx > 0 else 0.0
    return mobility, complexity


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd)
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    p = psd / np.sum(psd)
    return -np.sum(p * np.log2(p))

bonus_features = {
    'Log Detector': log_detector(x_rectified),
    'Hjorth Mobility': hjorth_parameters(x_rectified)[0],
    'Hjorth Complexity': hjorth_parameters(x_rectified)[1],
    'Spectral Entropy': spectral_entropy(psd_vals),
}

bonus_features
''')

add_markdown(r'''
## Machine Learning Extension

EMG features are often used as input to machine-learning models for activity recognition or prosthesis control. In a simple workflow:

1. create a feature matrix from windows of the signal;
2. assign labels (for example, active vs inactive);
3. split the data into train and test sets;
4. train a simple classifier and inspect the result.
''')

add_code(r'''
# Create a simple classification example using windowed features.
window_size = int(0.2 * sampling_frequency)
labels = df.get('activity_label', (np.abs(x_rectified) > np.median(np.abs(x_rectified))).astype(int))

feature_rows = []
for start in range(0, len(x_rectified) - window_size + 1, window_size):
    window = x_rectified[start:start + window_size]
    if len(window) < window_size:
        continue
    feature_rows.append({
        'MAV': mean_absolute_value(window),
        'RMS': root_mean_square(window),
        'WL': waveform_length(window),
        'ZC': zero_crossings(window),
        'Peak': peak_amplitude(window),
        'Label': labels.iloc[start] if hasattr(labels, 'iloc') else int(labels[start]),
    })

X = pd.DataFrame(feature_rows)
Y = X.pop('Label')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Classification accuracy:', round(accuracy_score(y_test, y_pred), 3))
print('Feature matrix shape:', X.shape)
X.head()
''')

add_markdown(r'''
## Conclusions

This notebook demonstrates a realistic EMG processing pipeline from raw data to feature extraction and interpretation. The main ideas are:

- EMG data need careful preprocessing before analysis;
- features can reveal muscle activation intensity and complexity;
- frequency-domain features help with fatigue-related interpretation;
- EMG features are widely used in biomedical engineering and machine-learning applications.

Students should remember that the quality of the recorded signal, the preprocessing choices, and the interpretation context all strongly affect the final conclusions.
''')

add_markdown(r'''
## Further Reading

- De Luca, C.J. (1997). The use of surface electromyography in biomechanics.
- Merletti, R. and Muceli, S. (2019). Tutorial. Surface EMG detection in space and time: surface EMG at the brain, muscle, and hand.
- Phinyomark, A. et al. (2012). A review of EMG feature extraction and application.
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
''')

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.11',
            'mimetype': 'text/x-python',
            'pygments_lexer': 'ipython3',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Notebook created at {out_path}')
