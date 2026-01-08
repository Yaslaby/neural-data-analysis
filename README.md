# Neural Data Analysis GUI

A Python application for visualizing and analyzing neural electrophysiology data, with a focus on hippocampal ripple detection from Open Ephys recordings.

Built for researchers who want to quickly load, preprocess, and analyze multi-channel neural recordings without writing code.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What It Does

- Loads Open Ephys `.continuous` files (single or multi-channel)
- Loads preprocessed data from `.mat` files (MATLAB format)
- Downsamples high-frequency recordings (30kHz → 1kHz) for faster processing
- Applies filters: 50Hz notch, bandpass (configurable)
- Shows before/after comparison of your preprocessing
- Loads sleep scoring `.mat` files to display sleep states (NREM, REM, Wake) along the recording
- Manual annotation system for marking events (ripples, spindles, artifacts)
- Exports annotations to CSV

---

## Demo Videos

### 1. Loading Data & Preprocessing

Load single or multiple channels, choose target frequency, and preprocess your data.

https://github.com/Yaslaby/neural-data-analysis/raw/main/videos/First%20step.mov

---

### 2. Navigation & Controls

Zoom in/out, adjust amplitude, resize panels, and select time windows.

https://github.com/Yaslaby/neural-data-analysis/raw/main/videos/Control%20bar.mov

---

### 3. Sleep Scoring Integration

Load sleep scoring from .mat files and overlay on your neural data.

https://github.com/Yaslaby/neural-data-analysis/raw/main/videos/Load%20sleep%20scoring.mov

---

### 4. Annotation System

Mark events manually using left-click drag, then export to CSV.

https://github.com/Yaslaby/neural-data-analysis/raw/main/videos/Annotation.mov

---

## Installation

### Requirements
- Python 3.8 or higher
- Works on Windows and macOS

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yaslaby/neural-data-analysis.git
cd neural-data-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
python main.py
```

### Dependencies
- PyQt5 (GUI framework)
- pyqtgraph (fast plotting)
- numpy, scipy (signal processing)
- MNE-Python (filtering and downsampling)
- matplotlib (optional, for some plots)

---

## How to Use

### Basic Workflow

1. **File → Open** - Load a `.continuous` file
2. Choose your target sampling rate (1000Hz is good for ripple analysis)
3. **Edit → Preprocess** - Apply filters
4. View the before/after comparison
5. **Edit → Load Sleep Scoring** - (Optional) Load sleep states from `.mat` file
6. Use left-click drag on the plot to annotate events
7. Export your annotations when done

### Loading Multiple Channels

Use **File → Load Multiple Channels** to load several `.continuous` files from the same recording session. They'll be synchronized automatically using timestamps.

### Loading Preprocessed Data (.mat files)

If you already have preprocessed data, use **File → Load Preprocessed Data (.mat)** to load it directly.

**What gets skipped for preprocessed data:**
- Downsampling (uses existing sampling rate)
- 50Hz notch filter (assumes noise already removed)

**What still applies:**
- Bandpass filter (80-250Hz for ripples, or your choice)

**Expected .mat file format:**
```matlab
data = [samples x channels];  % or 'signal', 'lfp', 'LFP'
fs = 1000;                    % or 'Fs', 'sampleRate', 'sample_rate'
```

### Annotations

- Left-click and drag on the plot to select a time region
- A dialog pops up to add description and category (ripple, spindle, delta, artifact)
- Double-click an annotation in the list to jump to that time
- Right-click to edit or delete
- Export to CSV for further analysis

---

## File Structure

```
├── main.py                 # Application entry point
├── dialogs.py              # UI dialog windows
├── PlotManager.py          # Handles plotting and zoom controls
├── MultiChannelLoader.py   # Loads multiple .continuous files
├── annotation.py           # Annotation system
├── comparison_view.py      # Before/after preprocessing view
├── workers.py              # Background processing threads
├── data_loader.py          # File loading utilities
├── sleep_scoring_mixin.py  # Sleep state visualization
├── sleep_scoring_dialog.py # Sleep scoring file loader
└── requirements.txt        # Python dependencies
```

---

## Preprocessing Pipeline

The preprocessing follows MNE-Python conventions:

1. **Downsampling** - Reduces sample rate using MNE's `resample()` with anti-aliasing
2. **Notch filter** - Removes 50Hz line noise (FIR, zero-phase)
3. **Bandpass filter** - Default 80-250Hz for ripple band (configurable)

Timestamps are preserved so you can relate detected events back to your original recording timeline.

---

## Supported File Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| Open Ephys | `.continuous` | Raw recordings from Open Ephys |
| MATLAB | `.mat` | Preprocessed data / Sleep scoring |

---

## Building Executables

The `.github/workflows/` folder contains GitHub Actions that automatically build:
- Windows `.exe`
- macOS `.app` bundle

Download standalone versions from the [Releases](https://github.com/Yaslaby/neural-data-analysis/releases) page.

---

## Known Limitations

- Only supports Open Ephys `.continuous` and MATLAB `.mat` formats
- Sleep scoring requires `.mat` files with a `states` variable
- Large files (>1GB) may be slow to load

---

## References

Ripple detection approach based on:
- Karlsson & Frank (2009) - Awake replay of remote experiences in the hippocampus. *Nature Neuroscience*

MNE-Python documentation:
- https://mne.tools/stable/index.html

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Genzel Lab, Donders Institute for Brain, Cognition and Behaviour, Radboud University

---

## Author

**Yasmine Shalaby** - Primary developer, Genzel Lab  
**Yixiao Zhang** - Supervisor, PhD candidate, Genzel Lab  
**Principal Investigator:** Prof. Lisa Genzel

Donders Institute for Brain, Cognition and Behaviour  
Radboud University, Nijmegen, The Netherlands

Questions or suggestions? Open an issue on GitHub.

---
