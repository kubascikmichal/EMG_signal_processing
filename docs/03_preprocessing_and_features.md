# EMG Preprocessing and Feature Extraction

This guide summarizes the main methods implemented in the notebook and in [emg_methods.py](../emg_methods.py).

## Preprocessing steps

### 1. DC offset removal

Subtracting the mean removes baseline drift and centers the signal around zero.

### 2. Band-pass filtering

A Butterworth band-pass filter between 20 Hz and 450 Hz preserves the main physiologically relevant EMG content while suppressing slow drift and high-frequency noise.

### 3. Notch filtering

A notch filter removes power-line interference, typically at 50 Hz or 60 Hz depending on the recording environment.

### 4. Rectification

Taking the absolute value makes activation bursts easier to interpret and prepares the signal for envelope extraction.

## Envelope methods

- Moving average envelope: simple smoothing over short windows
- RMS envelope: good for power-like amplitude estimation
- Hilbert envelope: smooth analytic-signal based envelope

## Time-domain features

Common features include:

- MAV: mean absolute value
- RMS: root mean square
- Variance and standard deviation
- IEMG: integrated EMG
- WL: waveform length
- ZC: zero crossings
- SSC: slope sign changes
- WAMP: Willison amplitude

## Frequency-domain features

These descriptors summarize how energy is distributed across frequencies:

- FFT magnitude spectrum
- PSD estimate
- mean frequency
- median frequency
- peak frequency
- total spectral power
