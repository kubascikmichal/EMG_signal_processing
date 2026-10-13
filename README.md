# EMG Signal Processing for Biomedical Engineering

This repository is an educational package for learning Electromyography (EMG) signal processing in Python. It combines a teaching notebook, implementation examples, and short markdown guides so students can study both the theory and the code.

## What students will learn

- how EMG signals are generated and why they are useful;
- how to load and explore raw biomedical data;
- how preprocessing improves signal quality;
- how to extract and interpret time-domain and frequency-domain features;
- how EMG features are used in practical applications such as prosthetic control, rehabilitation, and gesture recognition.

## Processing notebooks (real data)

| Notebook | Data |
|---|---|
| [notebooks/01_kaggle_hand_gestures_processing.ipynb](notebooks/01_kaggle_hand_gestures_processing.ipynb) | Kaggle [EMG Signal for Gesture Recognition](https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition): Myo armband, 8 forearm channels, 36 subjects, 6 hand movements |
| [notebooks/02_brainaccess_forearm.ipynb](notebooks/02_brainaccess_forearm.ipynb) | BrainAccess EMG: forearm flexors and extensors |
| [notebooks/02_brainaccess_arm.ipynb](notebooks/02_brainaccess_arm.ipynb) | BrainAccess EMG: upper arm (biceps brachii) |
| [notebooks/02_brainaccess_calf.ipynb](notebooks/02_brainaccess_calf.ipynb) | BrainAccess EMG: calf (gastrocnemius / soleus) |

All four notebooks share one pipeline: quality and sampling-rate check, DC removal, band-pass limited by the device's Nyquist frequency, mains notch, rectification, envelopes, activation detection, time- and frequency-domain features, windowed features with a classifier, and fatigue analysis. Notebook 01 downloads its dataset automatically; no Kaggle account is needed. For notebooks 02, copy the recordings into [data/brainaccess/](data/brainaccess/README.md).

## Repository structure

- [emg_io.py](emg_io.py) — dataset download and loaders (Kaggle gestures, BrainAccess `.fif` / `.edf` / `.csv`)
- [emg_plotting.py](emg_plotting.py) — shared plotting style and helpers
- [notebooks/emg_signal_processing_tutorial.ipynb](notebooks/emg_signal_processing_tutorial.ipynb) — the main teaching notebook
- [notebooks/emg_colab_example.ipynb](notebooks/emg_colab_example.ipynb) — a shorter Google Colab example for quick experimentation
- [emg_methods.py](emg_methods.py) — reusable implementation of preprocessing and feature extraction methods
- [docs/01_introduction.md](docs/01_introduction.md) — introduction to EMG for students
- [docs/02_dataset_sources.md](docs/02_dataset_sources.md) — public dataset sources and Kaggle guidance
- [docs/03_preprocessing_and_features.md](docs/03_preprocessing_and_features.md) — explanation of preprocessing and feature extraction
- [docs/04_emg_morphology.md](docs/04_emg_morphology.md) — reference guide for typical EMG signal morphology
- [requirements.txt](requirements.txt) — Python dependencies

## How to use this material

1. Create a Python environment and install the dependencies from [requirements.txt](requirements.txt).
2. Open [notebooks/emg_signal_processing_tutorial.ipynb](notebooks/emg_signal_processing_tutorial.ipynb) in VS Code or Jupyter.
3. Follow the markdown guides in [docs](docs) for conceptual background.
4. Use [emg_methods.py](emg_methods.py) as a reference implementation for the main methods.

## Recommended environment

Python 3.10 or newer is recommended.

## Notes

The notebook includes a reproducible fallback dataset so the workflow can still run even when Kaggle access is unavailable. The code also explains how to authenticate with the Kaggle API for real dataset download.
