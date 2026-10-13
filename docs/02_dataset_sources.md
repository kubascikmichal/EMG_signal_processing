# EMG Dataset Sources and Notes

The best educational EMG datasets are publicly available, contain raw or near-raw signals, and have a clearly documented sampling frequency.

## Recommended public sources

- Kaggle: search for terms such as `surface EMG`, `electromyography`, `sEMG`, or `muscle activity`.
- PhysioNet: useful for biomedical signal resources and related archives.
- GitHub repositories maintained by academic groups often share open data or example scripts.

## Datasets used in this repository

### Kaggle: EMG Signal for Gesture Recognition

- Slug: `sojanprajapati/emg-signal-for-gesture-recognition`, a mirror of UCI [EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures) (CC BY 4.0)
- Myo Thalmic armband, 8 forearm channels, 36 subjects × 2 runs
- Columns: `time` (ms), `channel1..8` (V), `class` (0 unmarked, 1 rest, 2 fist, 3 wrist flexion, 4 wrist extension, 5 radial deviation, 6 ulnar deviation, 7 extended palm), `label` (subject)
- **Pitfall:** rows are logged every ~1 ms, but the armband samples at 200 Hz, so most rows repeat the previous value. Resample to 200 Hz (`emg_io.resample_recording`) and keep the band-pass below 100 Hz.

### BrainAccess EMG recordings

Own recordings of the forearm, upper arm and calf. Format and folder layout: see [data/brainaccess/README.md](../data/brainaccess/README.md). BrainAccess devices typically sample at 250 Hz, which limits the usable band to ~20–112 Hz.

## What to look for in a dataset

A good teaching dataset should include:

- a clear sampling frequency;
- one or more EMG channels;
- time information or sample indexes;
- enough duration to observe activation bursts and rest periods;
- optional labels for activity or gesture classes.

## Kaggle API setup

If you want to download public Kaggle datasets, first create a Kaggle API token:

1. sign into Kaggle;
2. open your account settings;
3. download `kaggle.json`;
4. place it in `~/.kaggle/kaggle.json`.

Public datasets can also be downloaded without an account: `emg_io.download_kaggle_dataset` falls back to `https://www.kaggle.com/api/v1/datasets/download/<owner>/<dataset>` when no credentials are configured.

The notebook in this repository includes a fallback option so the workflow still works even when Kaggle access is unavailable.
