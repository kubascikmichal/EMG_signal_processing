"""Data loading helpers for the EMG notebooks.

* Kaggle "EMG data for gestures" (UCI #481 mirror, Myo armband, 8 forearm channels)
* BrainAccess recordings (MNE .fif saved by the BrainAccess SDK / Board, EDF, or CSV)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Kaggle hand-gesture dataset
# ---------------------------------------------------------------------------

KAGGLE_GESTURES_SLUG = "sojanprajapati/emg-signal-for-gesture-recognition"
GESTURES_CSV_NAME = "EMG-data.csv"
MYO_FS = 200.0  # nominal raw-EMG sampling rate of the Myo armband
GESTURE_NAMES: Dict[int, str] = {
    0: "unmarked",
    1: "rest",
    2: "fist",
    3: "wrist flexion",
    4: "wrist extension",
    5: "radial deviation",
    6: "ulnar deviation",
    7: "extended palm",
}
MYO_CHANNELS = [f"ch{i}" for i in range(1, 9)]


def _has_kaggle_credentials() -> bool:
    home = Path.home()
    return bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_KEY")
        or (home / ".kaggle" / "kaggle.json").exists()
        or (home / ".kaggle" / "access_token").exists()
    )


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive, refusing entries that would escape `out_dir`."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (out_dir / member).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise RuntimeError(f"Unsafe path in archive: {member}")
        zf.extractall(out_dir)


def download_kaggle_dataset(slug: str, out_dir: Path, expected_file: Optional[str] = None) -> Path:
    """Download and extract a public Kaggle dataset into `out_dir`.

    Uses the Kaggle CLI when API credentials are configured. Otherwise it falls
    back to Kaggle's public download endpoint, which works for public datasets
    without an account. Skips the download if `expected_file` already exists.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if expected_file and (out_dir / expected_file).exists():
        print(f"Dataset already present: {out_dir / expected_file}")
        return out_dir

    zip_path = out_dir / "archive.zip"
    if _has_kaggle_credentials() and shutil.which("kaggle"):
        print(f"Downloading {slug} with the Kaggle CLI ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)], check=True)
        zips = sorted(out_dir.glob("*.zip"))
        zip_path = zips[0]
    else:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
        print(f"No Kaggle credentials found - downloading public archive from {url} ...")
        with urllib.request.urlopen(url) as response, open(zip_path, "wb") as fh:
            shutil.copyfileobj(response, fh)

    _safe_extract(zip_path, out_dir)
    zip_path.unlink()
    print(f"Extracted to {out_dir}: {[p.name for p in out_dir.iterdir()]}")
    return out_dir


def load_gesture_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the gesture CSV and add a per-subject `recording` index.

    Columns of the original file: time (ms), channel1..channel8 (V), class, label (subject).
    Each subject performed the protocol twice; the time column restarts for the
    second run, which is how the two recordings are separated.
    """
    dtypes = {f"channel{i}": np.float32 for i in range(1, 9)}
    df = pd.read_csv(csv_path, dtype=dtypes)
    df = df.rename(columns={"time": "time_ms", "class": "gesture", "label": "subject"})
    df = df.rename(columns={f"channel{i}": f"ch{i}" for i in range(1, 9)})
    restart = df.groupby("subject")["time_ms"].diff().lt(0)
    df["recording"] = restart.astype(int).groupby(df["subject"]).cumsum()
    return df


def resample_recording(rec: pd.DataFrame, fs: float = MYO_FS) -> pd.DataFrame:
    """Resample one recording onto a uniform grid using zero-order hold.

    The CSV stores a row per logged millisecond, but the armband only delivers
    new samples in ~10 ms packets (nominal 200 Hz), so most rows repeat the
    previous value. Sampling the held value on a uniform `fs` grid recovers an
    evenly spaced signal that standard filters and spectra can be applied to.
    Amplitudes are converted from V to microvolts.
    """
    t = rec["time_ms"].to_numpy(dtype=float)
    grid = np.arange(t[0], t[-1], 1000.0 / fs)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    out = pd.DataFrame(rec[MYO_CHANNELS].to_numpy()[idx] * 1e6, columns=MYO_CHANNELS)
    out.insert(0, "t_s", (grid - grid[0]) / 1000.0)
    out["gesture"] = rec["gesture"].to_numpy()[idx]
    return out


# ---------------------------------------------------------------------------
# BrainAccess recordings
# ---------------------------------------------------------------------------

BRAINACCESS_EXTENSIONS = (".fif", ".fif.gz", ".edf", ".bdf", ".csv")
# Auxiliary channels written by the BrainAccess SDK that are not EMG.
NON_EMG_HINTS = ("sample", "accel", "gyro", "streaming", "digital", "marker", "trigger", "stim", "time", "annotation")


@dataclass
class EMGRecording:
    """A multichannel EMG recording in microvolts."""

    data: np.ndarray  # (n_channels, n_samples), microvolts
    fs: float
    ch_names: List[str]
    annotations: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["onset_s", "duration_s", "label"]))
    source: Optional[Path] = None

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.data.shape[1]) / self.fs

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs

    def channel(self, name_or_index) -> np.ndarray:
        idx = self.ch_names.index(name_or_index) if isinstance(name_or_index, str) else int(name_or_index)
        return self.data[idx]


def find_recordings(folder: Path) -> List[Path]:
    """List supported recording files in a folder (newest first)."""
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.is_file() and p.name.lower().endswith(BRAINACCESS_EXTENSIONS)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _is_emg_channel(name: str) -> bool:
    lname = name.lower()
    return not any(h in lname for h in NON_EMG_HINTS)


def _load_mne(path: Path, channels: Optional[Sequence[str]]) -> EMGRecording:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Reading .fif/.edf files requires MNE: pip install mne") from exc

    raw = mne.io.read_raw(path, preload=True, verbose="error")
    if channels is None:
        channels = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind not in ("stim", "misc") and _is_emg_channel(ch)]
        if not channels:  # some exports mark every channel as misc
            channels = [ch for ch in raw.ch_names if _is_emg_channel(ch)]
    data = raw.get_data(picks=list(channels)) * 1e6  # MNE stores volts

    ann = raw.annotations
    annotations = pd.DataFrame(
        {
            "onset_s": np.asarray(ann.onset) - raw.first_time,
            "duration_s": np.asarray(ann.duration),
            "label": [str(d) for d in ann.description],
        }
    )
    return EMGRecording(data=data, fs=float(raw.info["sfreq"]), ch_names=list(channels), annotations=annotations, source=path)


def _load_csv(path: Path, channels: Optional[Sequence[str]], fs: Optional[float], unit_scale: float) -> EMGRecording:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    time_col = next((lower[c] for c in ("time", "timestamp", "time_s", "t", "seconds") if c in lower), None)
    if fs is None:
        if time_col is None:
            raise ValueError(f"{path.name}: no time column found - pass fs= explicitly.")
        dt = float(np.median(np.diff(df[time_col].to_numpy(dtype=float))))
        # Timestamps are either seconds (dt ~ 0.004) or milliseconds (dt ~ 4).
        fs = round(1.0 / dt if dt < 0.5 else 1000.0 / dt, 3)

    label_col = next((lower[c] for c in ("annotation", "annotations", "marker", "markers", "event", "label") if c in lower), None)
    annotations = pd.DataFrame(columns=["onset_s", "duration_s", "label"])
    if label_col is not None:
        marks = df[label_col].fillna("").astype(str).str.strip()
        hit = ~marks.isin(["", "nan", "0", "0.0"])
        idx = np.flatnonzero(hit.to_numpy())
        annotations = pd.DataFrame({"onset_s": idx / fs, "duration_s": 0.0, "label": marks.iloc[idx].to_numpy()})

    if channels is None:
        channels = [
            c for c in df.columns
            if c not in (time_col, label_col) and _is_emg_channel(c) and pd.api.types.is_numeric_dtype(df[c])
        ]
    data = df[list(channels)].to_numpy(dtype=float).T * unit_scale
    return EMGRecording(data=data, fs=float(fs), ch_names=list(channels), annotations=annotations, source=path)


def load_brainaccess_recording(
    path: Path,
    channels: Optional[Sequence[str]] = None,
    fs: Optional[float] = None,
    csv_unit_scale: float = 1.0,
) -> EMGRecording:
    """Load a BrainAccess EMG recording.

    Supported formats
    -----------------
    * ``*-raw.fif`` - what ``brainaccess.utils.acquisition.EEG().data.save()`` and
      BrainAccess Board write (MNE format, volts, annotations included).
    * ``.edf`` / ``.bdf`` - read through MNE.
    * ``.csv`` - one column per channel, optional time column (s or ms) and an
      optional annotation/marker column. Values are assumed to be microvolts;
      use ``csv_unit_scale`` to convert (e.g. 1e6 if the CSV is in volts).

    Auxiliary channels (sample counter, accelerometer, streaming flag, ...) are
    dropped automatically unless `channels` is given.
    """
    path = Path(path)
    if path.name.lower().endswith((".fif", ".fif.gz", ".edf", ".bdf")):
        return _load_mne(path, channels)
    if path.suffix.lower() == ".csv":
        return _load_csv(path, channels, fs, csv_unit_scale)
    raise ValueError(f"Unsupported file type: {path.name} (expected one of {BRAINACCESS_EXTENSIONS})")
