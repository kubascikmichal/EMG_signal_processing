# BrainAccess EMG recordings

Put recordings into the folder of the recorded muscle:

| Folder | Muscle | Notebook |
|---|---|---|
| `forearm/` | forearm flexors / extensors | [02_brainaccess_forearm.ipynb](../../notebooks/02_brainaccess_forearm.ipynb) |
| `arm/` | upper arm — biceps brachii (optionally triceps) | [02_brainaccess_arm.ipynb](../../notebooks/02_brainaccess_arm.ipynb) |
| `calf/` | calf — gastrocnemius / soleus | [02_brainaccess_calf.ipynb](../../notebooks/02_brainaccess_calf.ipynb) |

By default each notebook processes the newest file in its folder. Set `RECORDING_FILE` in the configuration cell to pick another one.

## Supported formats

- `*-raw.fif`: saved by the BrainAccess Python SDK (`eeg.data.save(...)`) or BrainAccess Board. Sampling rate, channel names and annotations are read from the file.
- `.edf` / `.bdf`
- `.csv`: one column per channel in µV, an optional `time` column (seconds or milliseconds) and an optional `annotation` / `marker` column. Without a time column, set `CSV_FS` in the notebook.

Auxiliary channels (sample counter, accelerometer, streaming flag) are removed automatically. To process only some channels, list them in `CHANNELS`.

## Recording tips

- Mark protocol phases during recording with `eeg.annotate("fist")`, `eeg.annotate("rest")`, ... The markers appear in the plots and can be used as class labels (`LABEL_SOURCE = 'annotations'`).
- Start every recording with 5–10 s of complete rest. The burst detector estimates its baseline from the quietest seconds.
- For fatigue analysis (arm), include one sustained isometric contraction of 20–60 s.
