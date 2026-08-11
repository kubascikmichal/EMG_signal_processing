# EMG Dataset Sources and Notes

The best educational EMG datasets are publicly available, contain raw or near-raw signals, and have a clearly documented sampling frequency.

## Recommended public sources

- Kaggle: search for terms such as `surface EMG`, `electromyography`, `sEMG`, or `muscle activity`.
- PhysioNet: useful for biomedical signal resources and related archives.
- GitHub repositories maintained by academic groups often share open data or example scripts.

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

The notebook in this repository includes a fallback option so the workflow still works even when Kaggle access is unavailable.
