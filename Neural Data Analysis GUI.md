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
- Manual annotation system for marking events (ripples, spindles, artifacts)
- Exports annotations to CSV


---

## Screenshots

*[Add screenshots of your application here]*

---

## Installation

### Requirements
- Python 3.8 or higher
- Works on Windows and macOS

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-data-analysis.git
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
5. Use left-click drag on the plot to annotate events
6. Export your annotations when done

### Loading Preprocessed Data (.mat files)

If you already have preprocessed data (e.g., with custom noise removal), use **File → Load Preprocessed Data (.mat)** to load it directly.

**What gets skipped for preprocessed data:**
- Downsampling (uses existing sampling rate)
- 50Hz notch filter (assumes noise already removed)

**What still applies:**
- Bandpass filter (80-250Hz for ripples, or your choice)

This is useful when:
- Standard 50Hz notch doesn't work well (strong harmonics)
- You've applied custom preprocessing in MATLAB
- You want to compare different preprocessing methods

**Expected .mat file format:**
```matlab
% Your .mat file should contain:
data = [samples x channels];  % or 'signal', 'lfp', 'LFP'
fs = 1000;                    % or 'Fs', 'sampleRate', 'sample_rate'
% timestamps (optional) = [1 x samples];
```

The loader automatically detects common variable names. If it can't find your data, it will show available variables and let you specify the sampling rate.

### Loading Multiple Channels

Use **File → Load Multiple Channels** to load several `.continuous` files from the same recording session. They'll be synchronized automatically using timestamps.

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
├── data_loader.py          # File loading utilities (.continuous and .mat)
├── sleep_scoring_mixin.py  # Sleep state visualization
├── sleep_scoring_dialog.py # Sleep scoring file loader
└── requirements.txt        # Python dependencies
```

---

## Preprocessing Pipeline

The preprocessing follows MNE-Python conventions:

### For Raw Data (.continuous files)
1. **Downsampling** - Reduces sample rate using MNE's `resample()` with anti-aliasing
2. **Notch filter** - Removes 50Hz line noise (FIR, zero-phase)
3. **Bandpass filter** - Default 80-250Hz for ripple band (configurable)

### For Preprocessed Data (.mat files)
1. ~~Downsampling~~ - Skipped (uses existing rate)
2. ~~Notch filter~~ - Skipped (assumes already cleaned)
3. **Bandpass filter** - Applied for ripple detection

Timestamps are preserved so you can relate detected events back to your original recording timeline.

---

## Supported File Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| Open Ephys | `.continuous` | Raw recordings from Open Ephys |
| MATLAB | `.mat` | Preprocessed data from MATLAB |
| Sleep Scoring | `.mat` | Sleep state labels (separate loader) |

---

## For Developers

### Building Executables

The `.github/workflows/` folder contains GitHub Actions that automatically build:
- Windows `.exe`
- macOS `.app` bundle

These run on every release, so you can download standalone versions from the Releases page.

### Adding New Features

The code is modular. Key classes:
- `OpenEphysMainWindow` in `main.py` - Main application window
- `PlotManager` - All plotting logic
- `AnnotationManager` - Handles annotations and export
- `DataLoader` - File loading (mixin class)

---

## Known Limitations

- Only supports Open Ephys `.continuous` and MATLAB `.mat` formats (no `.nwb` or Neuralynx yet)
- Sleep scoring requires `.mat` files in a specific format
- Large files (>1GB) may be slow to load

---

## References

Ripple detection approach based on:
- Karlsson & Frank (2009) - Awake replay of remote experiences in the hippocampus

MNE-Python documentation:
- https://mne.tools/stable/index.html

---

## License

MIT License - Genzel Lab, Donders Institute, Radboud University

---

## Author

Yasmine Shalaby - Primary developer, Genzel Lab  
Yixiao Zhang - Supervisor, Genzel Lab

Questions or suggestions? Open an issue on GitHub.