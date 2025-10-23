#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from pathlib import Path

def load_data(file_path=None):
    if file_path and Path(file_path).exists():
        try:
            data = np.fromfile(file_path, dtype=np.int16)
            data = data.astype(np.float64) * 1e-6  # convert to microvolts
            return 20000, data
        except:
            pass
    raise FileNotFoundError(f"Data file not found or could not be read: {file_path}")

def main():
    # Load data
    file_path = "/Users/yaslaby/Documents/PyGt5_project/Channels/120_CH1.continuous"
    fs_orig, data = load_data(file_path)
    
    print(f"Loaded {len(data)} samples at {fs_orig} Hz")
    
    # Make MNE object
    info = mne.create_info(['LFP'], fs_orig, ['seeg'])
    raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
    
    # Downsample to 1kHz (MNE adds lowpass automatically)
    fs_new = 1000
    raw_downsampled = raw.copy().resample(fs_new, verbose=False)
    filtered_data = raw_downsampled.get_data()[0]
    
    print(f"Downsampled to {len(filtered_data)} samples at {fs_new} Hz")
    
    # Recreate MNE's filter parameters
    nyquist = fs_new / 2  # 500 Hz
    h_freq = nyquist  # cutoff frequency = nyquist frequency
    sfreq = fs_orig  # original sampling rate

    # Use MNE's actual transition bandwidth formula
    transition_width = min(max(h_freq * 0.25, 2.0), sfreq / 2.0 - h_freq)

    # MNE's filter length calculation (3.3 factor for Hamming window with firwin)
    filter_len = int(3.3 * sfreq / transition_width)
    if filter_len % 2 == 0:
        filter_len += 1

    # Create the filter that matches MNE
    h = signal.firwin(filter_len, h_freq, fs=sfreq, window='hamming')
    w, H = signal.freqz(h, worN=4096, fs=sfreq)

    # Set matplotlib style 
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create plots 
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(' ' \
    '', 
                 fontsize=14, fontweight='normal', y=0.95)
    
    # Time domain comparison
    ax = axes[0,0]
    t_orig = np.arange(len(data)) / fs_orig
    t_filt = np.arange(len(filtered_data)) / fs_new
    
    # Show first 0.05 seconds
    idx_orig = t_orig <= 0.05
    idx_filt = t_filt <= 0.05
    
    ax.plot(t_orig[idx_orig], data[idx_orig]*1e6, color='#2E86AB', alpha=0.7, 
            linewidth=1, label='Original signal')
    ax.plot(t_filt[idx_filt], filtered_data[idx_filt]*1e6, color='#A23B72', 
            linewidth=1.5, label='After MNE filtering')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Signal Before vs After Processing')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    # Frequency spectra
    ax = axes[0,1]
    
    # Compute FFTs
    f_orig = np.fft.fftfreq(len(data), 1/fs_orig)
    P_orig = np.abs(np.fft.fft(data))**2
    
    f_filt = np.fft.fftfreq(len(filtered_data), 1/fs_new)  
    P_filt = np.abs(np.fft.fft(filtered_data))**2
    
    # Plot up to 800 Hz
    mask1 = (f_orig >= 0) & (f_orig <= 800)
    mask2 = (f_filt >= 0) & (f_filt <= 500)
    
    ax.semilogy(f_orig[mask1], P_orig[mask1], color='#2E86AB', alpha=0.7, 
                linewidth=1, label='Original spectrum')
    ax.semilogy(f_filt[mask2], P_filt[mask2], color='#A23B72', linewidth=1.5, 
                label='Filtered spectrum')  
    ax.axvline(500, color='gray', linestyle='--', alpha=0.8, label='Nyquist limit (500 Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Frequency Content Comparison')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    # Filter response
    ax = axes[0,2]
    
    # Ideal vs actual filter
    f_ideal = np.linspace(0, 800, 800)
    H_ideal = np.ones_like(f_ideal)
    H_ideal[f_ideal > 500] = 0
    
    ax.plot(f_ideal, 20*np.log10(H_ideal + 1e-10), color='#F18F01', 
            linestyle='--', linewidth=2, label='Perfect brick-wall filter')
    ax.plot(w[w<=800], 20*np.log10(np.abs(H[w<=800])), color='#A23B72', 
            linewidth=2, label="MNE's actual filter")
    ax.axvline(500, color='gray', linestyle=':', alpha=0.8, label='Cutoff frequency')
    ax.axhline(-3, color='orange', linestyle=':', alpha=0.8, label='-3 dB point')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Filter Response: Ideal vs Reality')
    ax.set_ylim(-65, 5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Transition region detail
    ax = axes[1,0]
    transition_mask = (w >= 400) & (w <= 600)
    ax.plot(w[transition_mask], 20*np.log10(np.abs(H[transition_mask])), 
            color='#A23B72', linewidth=2.5)
    ax.axvline(500, color='gray', linestyle='--', alpha=0.8, label='Target cutoff')
    ax.axhline(-3, color='orange', linestyle='--', alpha=0.8, label='-3 dB')
    ax.fill_between([450, 550], -45, 5, alpha=0.15, color='#F18F01', 
                   label=f'Transition band (~{transition_width:.0f} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Why Filters Aren\'t Perfect Walls')
    ax.set_ylim(-45, 5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Passband ripple
    ax = axes[1,1]
    passband_mask = (w >= 0) & (w <= 400)
    passband_dB = 20*np.log10(np.abs(H[passband_mask]))
    ax.plot(w[passband_mask], passband_dB, color='#2E86AB', linewidth=2)
    ax.axhline(0, color='green', linestyle='--', alpha=0.7, label='Perfect response (0 dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Passband Ripple (Should Be Flat)')
    ax.set_ylim(-0.6, 0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Stopband attenuation
    ax = axes[1,2]
    stopband_mask = (w >= 600) & (w <= 800)
    if np.any(stopband_mask):
        stopband_dB = 20*np.log10(np.abs(H[stopband_mask]))
        ax.plot(w[stopband_mask], stopband_dB, color='#C73E1D', linewidth=2)
        ax.axhline(-40, color='orange', linestyle='--', alpha=0.8, label='-40 dB (good)')
        ax.axhline(-60, color='green', linestyle='--', alpha=0.8, label='-60 dB (excellent)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Stopband Attenuation')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-80, -20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Analysis summary
    print("\n" + "="*50)
    print("FILTER ANALYSIS SUMMARY")
    print("="*50)
    
    # Find -3dB point
    db_response = 20*np.log10(np.abs(H))
    cutoff_3db_idx = np.argmin(np.abs(db_response + 3))
    cutoff_3db = w[cutoff_3db_idx]
    
    # Passband ripple
    ripple_db = np.max(passband_dB) - np.min(passband_dB)
    
    # Stopband suppression
    if np.any(stopband_mask):
        max_stopband = np.max(stopband_dB)
        stopband_str = f"{max_stopband:.1f} dB"
    else:
        stopband_str = "insufficient frequency range"
    
    print(f"Target cutoff frequency: {nyquist:.0f} Hz (Nyquist)")
    print(f"Actual -3dB cutoff: {cutoff_3db:.1f} Hz")
    print(f"Transition bandwidth: {transition_width:.1f} Hz")
    print(f"Filter length: {filter_len} samples ({filter_len/fs_orig*1000:.1f} ms)")
    print(f"Passband ripple: {ripple_db:.3f} dB") 
    print(f"Stopband attenuation: {stopband_str}")
    
    print(f"\nMNE's transition bandwidth formula:")
    print(f"  min(max(cutoff * 0.25, 2.0), sfreq/2 - cutoff)")
    print(f"  = min(max({h_freq} * 0.25, 2.0), {sfreq}/2 - {h_freq})")
    print(f"  = min(max({h_freq * 0.25}, 2.0), {sfreq/2 - h_freq})")
    print(f"  = {transition_width:.1f} Hz")
    
    print("\n" + "-"*50)
    print("WHY FILTERS AREN'T PERFECT BRICK WALLS:")
    print("-"*50)
    print("1. Finite length: Real filters can't be infinitely long")
    print("2. Gibbs phenomenon: Sharp cutoffs create ringing artifacts")  
    print("3. Window trade-offs: Smooth windows reduce ripple but widen transitions")
    print("4. Computational limits: Longer filters = more processing time")
    print("5. Causality: Real-time filters can't see future samples")
    
    print("\n" + "-"*50)
    print("WHAT MNE ACTUALLY DOES:")
    print("-"*50)
    print("• Uses FIR filter with Hamming window (good compromise)")
    print("• Applies zero-phase filtering (no time delay)")
    print("• Chooses filter length based on transition bandwidth")
    print("• Balances filter quality vs computational efficiency")
    print("• Follows established neurophysiology standards")
    print("• Provides adequate anti-aliasing for typical brain signals")

if __name__ == "__main__":
    main()