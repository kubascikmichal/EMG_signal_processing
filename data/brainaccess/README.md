# BrainAccess EMG recordings

Put recordings into the folder of the recorded muscle:

| Folder | Muscle | Notebook |
|---|---|---|
| `forearm/` | forearm flexors / extensors | [02_brainaccess_forearm.ipynb](../../notebooks/02_brainaccess_forearm.ipynb) |
| `arm/` | upper arm — biceps brachii (optionally triceps) | [02_brainaccess_arm.ipynb](../../notebooks/02_brainaccess_arm.ipynb) |
| `calf/` | calf — gastrocnemius / soleus | [02_brainaccess_calf.ipynb](../../notebooks/02_brainaccess_calf.ipynb) |

Put **one file per task** and name it without personal data: `task-1.csv`, `task-2.csv`, ... (no subject ID, date, device serial or session UUID). Each notebook walks through one task step by step (the first one, or the file set in `TASK`) and then processes all tasks in the folder and compares them (Section 11).

`forearm/` contains four tasks recorded with BrainAccess MINI (8 EMG channels, 500 Hz), numbered in recording order.

## Supported formats

- `*-raw.fif`: saved by the BrainAccess Python SDK (`eeg.data.save(...)`) or BrainAccess Board. Sampling rate, channel names and annotations are read from the file.
- `.edf` / `.bdf`
- `.csv`: one column per channel, an optional `time` column (seconds or milliseconds) and an optional `annotation` / `marker` column. Without a time column, set `CSV_FS` in the notebook. BrainAccess CSV exports are in volts. The notebooks convert them with `CSV_UNIT_SCALE = 1e6`; set it to `1.0` for a CSV that is already in µV.

Auxiliary channels (index column, sample counter, accelerometer, electrode-contact flags, battery, streaming flag) are removed automatically. Rows with `Streaming = 0` are packets lost in transmission and are filled with zeros in the export; the loader interpolates them, and bursts and windows that touch them are excluded from the features. To process only some channels, list them in `CHANNELS`.

## Recording tips

- Mark protocol phases during recording with `eeg.annotate("fist")`, `eeg.annotate("rest")`, ... The markers appear in the plots and can be used as class labels (`LABEL_SOURCE = 'annotations'`).
- Start every recording with 5–10 s of complete rest. The burst detector estimates its baseline from the quietest seconds.
- For fatigue analysis (arm), include one sustained isometric contraction of 20–60 s.
