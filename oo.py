#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal

def demonstrate_antialiasing():
    """Show exactly where and why the anti-aliasing filter is needed."""
    
    # Create test signal with frequencies above Nyquist
    fs_orig = 2000  # 2 kHz original
    fs_new = 500    # 500 Hz target (Nyquist = 250 Hz)
    t = np.linspace(0, 2, fs_orig * 2)
    
    # Signal with frequencies both below AND above new Nyquist
    signal_clean = np.sin(2*np.pi*100*t)     # 100 Hz - safe
    signal_alias = np.sin(2*np.pi*400*t)     # 400 Hz - will alias to 100 Hz!
    signal_combined = signal_clean + signal_alias
    
    print("Demonstration: Why Anti-Aliasing Filters Are Essential")
    print("="*55)
    print(f"Original sampling rate: {fs_orig} Hz")
    print(f"Target sampling rate: {fs_new} Hz") 
    print(f"New Nyquist frequency: {fs_new/2} Hz")
    print(f"Test signal contains: 100 Hz (safe) + 400 Hz (dangerous!)")
    print()
    
    # Method 1: Naive downsampling (NO anti-aliasing filter)
    print("METHOD 1: Naive downsampling (BAD - causes aliasing)")
    print("-" * 50)
    
    # Just decimate without filtering - THIS IS WRONG
    naive_decimated = signal_combined[::4]  # Take every 4th sample
    t_decimated = np.linspace(0, 2, len(naive_decimated))
    
    # Method 2: MNE's proper resampling (WITH anti-aliasing filter)
    print("METHOD 2: MNE's proper resampling (GOOD - prevents aliasing)")
    print("-" * 60)
    
    info = mne.create_info(['test'], fs_orig, ['misc'])
    raw = mne.io.RawArray(signal_combined.reshape(1, -1), info, verbose=False)
    
    # This line contains the anti-aliasing filter!
    raw_resampled = raw.copy().resample(fs_new, verbose=True)
    proper_decimated = raw_resampled.get_data()[0]
    t_proper = np.linspace(0, 2, len(proper_decimated))
    
    # Method 3: Manual anti-aliasing (showing what MNE does internally)
    print("\nMETHOD 3: Manual anti-aliasing (showing MNE's internals)")
    print("-" * 58)
    
    nyquist_new = fs_new / 2  # 250 Hz
    
    # Create the same filter MNE uses
    h_freq = nyquist_new
    transition_width = min(max(h_freq * 0.25, 2.0), fs_orig/2 - h_freq)
    filter_len = int(3.3 * fs_orig / transition_width)
    if filter_len % 2 == 0:
        filter_len += 1
    
    print(f"Anti-aliasing filter specs:")
    print(f"  Cutoff frequency: {h_freq} Hz")
    print(f"  Transition bandwidth: {transition_width:.1f} Hz") 
    print(f"  Filter length: {filter_len} samples")
    
    # Apply the anti-aliasing filter manually
    aa_filter = signal.firwin(filter_len, h_freq, fs=fs_orig, window='hamming')
    filtered_signal = signal.filtfilt(aa_filter, 1, signal_combined)
    
    # Then decimate
    manual_decimated = filtered_signal[::4]
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Anti-Aliasing Filter: Where It Lives and Why We Need It', 
                 fontsize=14, fontweight='bold')
    
    # Time domain comparison
    ax = axes[0, 0]
    ax.plot(t[:1000], signal_combined[:1000], 'b-', label='Original (100Hz + 400Hz)', linewidth=1.5)
    ax.plot(t_decimated[:250], naive_decimated[:250], 'r-', label='Naive decimation', linewidth=2, alpha=0.8)
    ax.plot(t_proper[:250], proper_decimated[:250], 'g-', label='MNE resampling', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Time Domain: Original vs Downsampled')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    
    # Frequency domain - original signal
    ax = axes[0, 1]
    f_orig = np.fft.fftfreq(len(signal_combined), 1/fs_orig)
    P_orig = np.abs(np.fft.fft(signal_combined))**2
    
    pos_freq = f_orig >= 0
    ax.semilogy(f_orig[pos_freq], P_orig[pos_freq], 'b-', linewidth=2, label='Original spectrum')
    ax.axvline(100, color='green', linestyle='--', alpha=0.7, label='100 Hz (safe)')
    ax.axvline(400, color='red', linestyle='--', alpha=0.7, label='400 Hz (will alias!)')
    ax.axvline(250, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='New Nyquist (250 Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Original Signal Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)
    
    # Naive decimation spectrum (shows aliasing!)
    ax = axes[1, 0]
    f_naive = np.fft.fftfreq(len(naive_decimated), 1/fs_new)
    P_naive = np.abs(np.fft.fft(naive_decimated))**2
    
    pos_freq_naive = f_naive >= 0
    ax.semilogy(f_naive[pos_freq_naive], P_naive[pos_freq_naive], 'r-', linewidth=2, label='Naive decimation')
    ax.axvline(100, color='green', linestyle='--', alpha=0.7, label='100 Hz (original)')
    ax.axvline(100, color='red', linestyle=':', linewidth=3, alpha=0.7, label='400→100 Hz (aliased!)')
    ax.axvline(250, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Nyquist limit')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('BAD: Naive Decimation (400 Hz aliases to 100 Hz!)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    
    # Proper MNE resampling spectrum
    ax = axes[1, 1]
    f_proper = np.fft.fftfreq(len(proper_decimated), 1/fs_new)
    P_proper = np.abs(np.fft.fft(proper_decimated))**2
    
    pos_freq_proper = f_proper >= 0
    ax.semilogy(f_proper[pos_freq_proper], P_proper[pos_freq_proper], 'g-', linewidth=2, label='MNE resampling')
    ax.axvline(100, color='green', linestyle='--', alpha=0.7, label='100 Hz (preserved)')
    ax.axvline(250, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Nyquist limit')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('GOOD: MNE Resampling (400 Hz removed!)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    
    # Show the anti-aliasing filter response
    ax = axes[2, 0]
    w, H = signal.freqz(aa_filter, worN=4096, fs=fs_orig)
    ax.plot(w, 20*np.log10(np.abs(H)), 'purple', linewidth=2, label='Anti-aliasing filter')
    ax.axvline(250, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Cutoff (250 Hz)')
    ax.axvline(100, color='green', linestyle='--', alpha=0.7, label='100 Hz (passes)')
    ax.axvline(400, color='red', linestyle='--', alpha=0.7, label='400 Hz (blocked)')
    ax.axhline(-3, color='gray', linestyle=':', alpha=0.7, label='-3 dB')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('The Anti-Aliasing Filter Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 5)
    
    # Show where aliasing would occur
    ax = axes[2, 1]
    freq_range = np.linspace(0, 1000, 1000)
    aliased_freq = np.mod(freq_range, fs_new)
    aliased_freq[aliased_freq > fs_new/2] = fs_new - aliased_freq[aliased_freq > fs_new/2]
    
    ax.plot(freq_range, aliased_freq, 'r-', linewidth=2, label='Where frequencies end up')
    ax.axhline(100, color='green', linestyle='--', alpha=0.7, label='100 Hz target')
    ax.axvline(400, color='red', linestyle='--', alpha=0.7, label='400 Hz source')
    ax.axvline(250, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Nyquist limit')
    ax.plot(400, 100, 'ro', markersize=10, label='400 Hz → 100 Hz (aliasing!)')
    ax.set_xlabel('Original Frequency (Hz)')
    ax.set_ylabel('Apparent Frequency After Decimation (Hz)')
    ax.set_title('Aliasing Map: Where High Frequencies Go')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 250)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY: WHERE THE ANTI-ALIASING FILTER LIVES")
    print("="*60)
    print("1. INSIDE MNE's resample() method")
    print("   - You don't see it, but it's automatically applied")
    print("   - Happens BEFORE decimation")
    print()
    print("2. The filter we recreated shows what MNE did:")
    print(f"   - Low-pass filter with {h_freq} Hz cutoff")
    print(f"   - Removes all frequencies above {h_freq} Hz")
    print(f"   - Prevents aliasing when downsampling to {fs_new} Hz")
    print()
    print("3. Without this filter:")
    print("   - High frequencies would fold back (alias)")
    print("   - 400 Hz would appear as 100 Hz")
    print("   - Data would be corrupted and unusable")
    print()
    print("4. The recreation code shows MNE's internal process:")
    print("   - Same filter design (firwin + Hamming window)")
    print("   - Same transition bandwidth formula")
    print("   - Same filter length calculation")

if __name__ == "__main__":
    demonstrate_antialiasing()