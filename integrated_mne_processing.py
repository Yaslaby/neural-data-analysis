"""
Enhanced MNE Processing Pipeline for Neural Data Analysis
Provides flexible signal processing with proper error handling and validation
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.filter import notch_filter
from scipy import signal
from pathlib import Path

def process_for_ripples_mne_standard(data, fs_orig=20000, fs_target=None, 
                                    apply_notch=True, apply_bandpass=True, 
                                    bandpass_low=100, bandpass_high=250, show_plot=False):
    """
    Enhanced MNE processing pipeline for neural data
    
    Parameters:
    -----------
    data : array-like
        Input neural data (1D array)
    fs_orig : float
        Original sampling frequency (default: 20000)
    fs_target : float or None
        Target sampling frequency. If None, no resampling
    apply_notch : bool
        Apply 50Hz notch filter (default: True)
    apply_bandpass : bool
        Apply bandpass filter (default: True)
    bandpass_low : float
        Bandpass low cutoff (default: 100)
    bandpass_high : float
        Bandpass high cutoff (default: 250)
    show_plot : bool
        Show processing visualization (default: False)
    
    Returns:
    --------
    dict : Processing results with all stages
    """
    
    # Input validation
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("Data must be 1D array")
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    if fs_orig <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    print(f"🔄 MNE Processing: {len(data)} samples @ {fs_orig}Hz")
    
    # Step 1: Create MNE Raw object
    try:
        info = mne.create_info(['LFP'], fs_orig, ['seeg'])
        raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
        print("✓ MNE Raw object created")
    except Exception as e:
        raise RuntimeError(f"Failed to create MNE Raw object: {e}")
    
    # Step 2: Resampling
    if fs_target is not None and fs_target != fs_orig:
        if fs_target <= 0:
            raise ValueError("Target sampling frequency must be positive")
        print(f"🔄 Resampling: {fs_orig}Hz → {fs_target}Hz")
        try:
            raw_resampled = raw.copy().resample(fs_target, verbose=False)
            current_fs = fs_target
            print("✓ Resampling complete")
        except Exception as e:
            raise RuntimeError(f"Resampling failed: {e}")
    else:
        raw_resampled = raw.copy()
        current_fs = fs_orig
        print(f"✓ No resampling - keeping {fs_orig}Hz")
    
    # Step 3: Notch filtering
    if apply_notch:
        nyquist = current_fs / 2
        if nyquist <= 50:
            print(f"⚠️  Skipping notch filter: Nyquist ({nyquist}Hz) ≤ 50Hz")
            raw_notched = raw_resampled.copy()
        else:
            print("🔄 Applying 50Hz notch filter")
            try:
                raw_notched = raw_resampled.copy()
                raw_notched._data[0] = notch_filter(
                    x=raw_resampled._data[0],
                    Fs=current_fs,
                    freqs=[50],
                    filter_length='auto',
                    method='fir',
                    phase='zero',
                    verbose=False
                )
                print("✓ Notch filter applied")
            except Exception as e:
                print(f"⚠️  Notch filter failed: {e}, continuing without")
                raw_notched = raw_resampled.copy()
    else:
        raw_notched = raw_resampled.copy()
        print("✓ Notch filter skipped")
    
    # Step 4: Bandpass filtering
    if apply_bandpass:
        # Validate and adjust filter parameters
        nyquist = current_fs / 2
        original_low, original_high = bandpass_low, bandpass_high
        
        if bandpass_high >= nyquist:
            bandpass_high = nyquist * 0.9
            print(f"⚠️  Adjusted high cutoff: {original_high}Hz → {bandpass_high:.1f}Hz")
        
        if bandpass_low >= bandpass_high:
            bandpass_low = bandpass_high * 0.3
            print(f"⚠️  Adjusted low cutoff: {original_low}Hz → {bandpass_low:.1f}Hz")
        
        if bandpass_low < 1:
            bandpass_low = 1
            print(f"⚠️  Minimum low cutoff: {bandpass_low}Hz")
        
        print(f"🔄 Bandpass filter: {bandpass_low:.1f}-{bandpass_high:.1f}Hz")
        
        try:
            # Calculate reasonable transition bandwidths
            passband_width = bandpass_high - bandpass_low
            l_trans = max(1, min(bandpass_low * 0.2, passband_width * 0.1))
            h_trans = max(1, min((nyquist - bandpass_high) * 0.5, passband_width * 0.1))
            
            raw_filtered = raw_notched.copy().filter(
                l_freq=bandpass_low,
                h_freq=bandpass_high,
                l_trans_bandwidth=l_trans,
                h_trans_bandwidth=h_trans,
                method='fir',
                phase='zero',
                fir_window='hamming',
                verbose=False
            )
            print("✓ Bandpass filter applied")
        except Exception as e:
            raise RuntimeError(f"Bandpass filtering failed: {e}")
    else:
        raw_filtered = raw_notched.copy()
        print("✓ Bandpass filter skipped")
    
    # Step 5: Hilbert transform
    print("🔄 Computing Hilbert envelope")
    try:
        raw_hilbert = raw_filtered.copy()
        raw_hilbert.apply_hilbert(envelope=False, verbose=False)
        envelope = np.abs(raw_hilbert._data[0])
        print("✓ Hilbert transform complete")
    except Exception as e:
        raise RuntimeError(f"Hilbert transform failed: {e}")
    
    # Compile results
    results = {
        'original': raw._data[0],
        'resampled': raw_resampled._data[0],
        'notched': raw_notched._data[0],
        'ripple_filtered': raw_filtered._data[0],  # Main output
        'analytic_signal': raw_hilbert._data[0],
        'envelope': envelope,
        'fs_orig': fs_orig,
        'fs_target': fs_target,
        'fs_current': current_fs,
        'processing_params': {
            'apply_notch': apply_notch,
            'apply_bandpass': apply_bandpass,
            'bandpass_low': bandpass_low,
            'bandpass_high': bandpass_high
        }
    }
    
    print(f"✅ Processing complete: {len(results['ripple_filtered'])} samples @ {current_fs}Hz")
    
    # Show visualization if requested
    if show_plot:
        create_mne_processing_plot(results)
    
    return results

def detect_ripples_mne_standard(envelope, fs=1000, params=None):
    """
    Detect ripple events using amplitude envelope
    
    Parameters:
    -----------
    envelope : array-like
        Signal envelope from Hilbert transform
    fs : float
        Sampling frequency of envelope
    params : dict or None
        Detection parameters
    
    Returns:
    --------
    dict : Detection results
    """
    
    if params is None:
        params = {
            'threshold': 3.0,       # Standard deviations above mean
            'min_duration': 0.015,  # 15ms minimum
            'max_duration': 0.200,  # 200ms maximum
            'min_interval': 0.050   # 50ms between ripples
        }
    
    envelope = np.asarray(envelope)
    if len(envelope) == 0:
        return {'ripple_events': [], 'envelope': envelope, 'threshold': 0}
    
    # Calculate detection threshold
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    threshold = mean_env + params['threshold'] * std_env
    
    # Find threshold crossings
    above_threshold = envelope > threshold
    
    if not np.any(above_threshold):
        return {
            'ripple_events': [],
            'envelope': envelope,
            'threshold': threshold,
            'detection_params': params,
            'sampling_rate': fs
        }
    
    # Find event boundaries
    diff_above = np.diff(above_threshold.astype(int))
    starts = np.where(diff_above == 1)[0] + 1
    ends = np.where(diff_above == -1)[0] + 1
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.concatenate([[0], starts])
    if above_threshold[-1]:
        ends = np.concatenate([ends, [len(above_threshold)]])
    
    # Apply duration and interval criteria
    ripple_events = []
    min_duration_samples = int(params['min_duration'] * fs)
    max_duration_samples = int(params['max_duration'] * fs)
    min_interval_samples = int(params['min_interval'] * fs)
    
    for start, end in zip(starts, ends):
        duration_samples = end - start
        duration_sec = duration_samples / fs
        
        # Check duration criteria
        if min_duration_samples <= duration_samples <= max_duration_samples:
            # Find peak within event
            peak_idx = start + np.argmax(envelope[start:end])
            peak_amplitude = envelope[peak_idx]
            
            ripple_event = {
                'start_time': start / fs,
                'end_time': end / fs,
                'duration': duration_sec,
                'peak_time': peak_idx / fs,
                'peak_amplitude': peak_amplitude,
                'start_sample': start,
                'end_sample': end,
                'peak_sample': peak_idx
            }
            
            # Check minimum interval from previous event
            if (not ripple_events or 
                (start - ripple_events[-1]['end_sample']) >= min_interval_samples):
                ripple_events.append(ripple_event)
    
    return {
        'ripple_events': ripple_events,
        'envelope': envelope,
        'threshold': threshold,
        'detection_params': params,
        'sampling_rate': fs
    }

def create_mne_processing_plot(results):
    """Create comprehensive visualization of processing pipeline"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('MNE Processing Pipeline Results', fontsize=16, fontweight='bold')
    
    fs_orig = results['fs_orig']
    fs_current = results['fs_current']
    
    # Limit samples for plotting performance
    max_plot_samples = 10000
    n_orig = min(max_plot_samples, len(results['original']))
    n_proc = min(max_plot_samples, len(results['ripple_filtered']))
    
    t_orig = np.arange(n_orig) / fs_orig
    t_proc = np.arange(n_proc) / fs_current
    
    # Plot 1: Original vs Resampled
    axes[0,0].plot(t_orig, results['original'][:n_orig] * 1e6, 'b-', alpha=0.7, 
                   label=f'Original ({fs_orig}Hz)')
    if len(results['resampled']) > 0:
        n_resamp = min(max_plot_samples, len(results['resampled']))
        t_resamp = np.arange(n_resamp) / fs_current
        axes[0,0].plot(t_resamp, results['resampled'][:n_resamp] * 1e6, 'g-', 
                       label=f'Resampled ({fs_current}Hz)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude (µV)')
    axes[0,0].set_title('Step 1-2: Raw Data → Resampling')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Resampled vs Notched
    if len(results['resampled']) > 0 and len(results['notched']) > 0:
        axes[0,1].plot(t_proc, results['resampled'][:n_proc] * 1e6, 'g-', alpha=0.7, label='Resampled')
        axes[0,1].plot(t_proc, results['notched'][:n_proc] * 1e6, 'r-', label='50Hz Notched')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Amplitude (µV)')
        axes[0,1].set_title('Step 3: Notch Filtering')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Notched vs Bandpass
    axes[1,0].plot(t_proc, results['notched'][:n_proc] * 1e6, 'r-', alpha=0.7, label='Notched')
    axes[1,0].plot(t_proc, results['ripple_filtered'][:n_proc] * 1e6, 'purple', label='Bandpass Filtered')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Amplitude (µV)')
    
    # Add filter parameters to title
    params = results.get('processing_params', {})
    if params.get('apply_bandpass'):
        filter_range = f"{params.get('bandpass_low', '?')}-{params.get('bandpass_high', '?')}Hz"
        axes[1,0].set_title(f'Step 4: Bandpass Filter ({filter_range})')
    else:
        axes[1,0].set_title('Step 4: Bandpass Filter (skipped)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Filtered Signal vs Envelope
    axes[1,1].plot(t_proc, results['ripple_filtered'][:n_proc] * 1e6, 'purple', 
                   alpha=0.5, label='Filtered Signal')
    axes[1,1].plot(t_proc, results['envelope'][:n_proc] * 1e6, 'orange', 
                   linewidth=2, label='Hilbert Envelope')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Amplitude (µV)')
    axes[1,1].set_title('Step 5: Hilbert Transform')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5-6: Frequency domain analysis
    try:
        # Original spectrum
        nperseg_orig = min(4096, len(results['original']) // 4)
        f_orig, psd_orig = signal.welch(results['original'], fs=fs_orig, nperseg=nperseg_orig)
        
        mask_orig = f_orig <= min(500, fs_orig/2 * 0.95)
        axes[2,0].semilogy(f_orig[mask_orig], psd_orig[mask_orig], 'b-', alpha=0.8, label='Original PSD')
        axes[2,0].axvline(50, color='red', linestyle='--', alpha=0.7, label='50Hz')
        
        # Highlight filter band
        params = results.get('processing_params', {})
        if params.get('apply_bandpass'):
            low = params.get('bandpass_low', 100)
            high = params.get('bandpass_high', 250)
            axes[2,0].axvspan(low, high, alpha=0.2, color='purple', label=f'Filter Band')
        
        axes[2,0].set_xlabel('Frequency (Hz)')
        axes[2,0].set_ylabel('Power Spectral Density')
        axes[2,0].set_title('Original Frequency Content')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Filtered spectrum
        nperseg_filt = min(1024, len(results['ripple_filtered']) // 4)
        f_filt, psd_filt = signal.welch(results['ripple_filtered'], fs=fs_current, nperseg=nperseg_filt)
        
        mask_filt = f_filt <= min(300, fs_current/2 * 0.95)
        axes[2,1].semilogy(f_filt[mask_filt], psd_filt[mask_filt], 'purple', 
                          linewidth=2, label='Filtered PSD')
        
        # Show actual passband
        if params.get('apply_bandpass'):
            low = params.get('bandpass_low', 100)
            high = params.get('bandpass_high', 250)
            axes[2,1].axvspan(low, high, alpha=0.3, color='purple', label=f'Passband ({low}-{high}Hz)')
        
        axes[2,1].set_xlabel('Frequency (Hz)')
        axes[2,1].set_ylabel('Power Spectral Density')
        axes[2,1].set_title('Filtered Frequency Content')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Warning: Could not create frequency plots: {e}")
        for ax in [axes[2,0], axes[2,1]]:
            ax.text(0.5, 0.5, 'Frequency analysis\nunavailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()
    return fig

# Utility functions
def load_data(file_path=None):
    """Simple data loading utility"""
    if file_path and Path(file_path).exists():
        try:
            data = np.fromfile(file_path, dtype=np.int16)
            return 20000, data.astype(np.float64) * 1e-6
        except Exception as e:
            raise IOError(f"Could not read file {file_path}: {e}")
    raise FileNotFoundError(f"File not found: {file_path}")

# Backward compatibility functions
def process_for_ripples(data, fs_orig=20000, fs_target=1000):
    """Legacy function for backward compatibility"""
    return process_for_ripples_mne_standard(
        data, fs_orig=fs_orig, fs_target=fs_target, show_plot=False
    )

def create_processing_plot(results):
    """Legacy function for backward compatibility"""
    return create_mne_processing_plot(results)

def main():
    """Example usage and testing"""
    print("MNE Processing Pipeline - Example Usage")
    print("="*50)
    
    # Create test data
    fs = 20000
    duration = 2.0  # 2 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Simulate neural data with multiple frequency components
    np.random.seed(42)
    signal_data = (
        np.random.normal(0, 20, len(t)) +  # Background noise
        50 * np.sin(2 * np.pi * 8 * t) +   # Theta (8 Hz)
        20 * np.sin(2 * np.pi * 50 * t) +  # Line noise (50 Hz)
        30 * np.sin(2 * np.pi * 150 * t)   # Ripple frequency (150 Hz)
    )
    
    print(f"Test data: {len(signal_data)} samples @ {fs}Hz")
    
    try:
        # Process with visualization
        results = process_for_ripples_mne_standard(
            signal_data,
            fs_orig=fs,
            fs_target=1000,
            apply_notch=True,
            apply_bandpass=True,
            bandpass_low=100,
            bandpass_high=200,
            show_plot=True
        )
        
        # Test ripple detection
        detection_results = detect_ripples_mne_standard(results['envelope'], fs=1000)
        
        print(f"\nResults:")
        print(f"• Original data: {len(results['original'])} samples")
        print(f"• Processed data: {len(results['ripple_filtered'])} samples")
        print(f"• Detected ripples: {len(detection_results['ripple_events'])}")
        print("✅ Test completed successfully!")
        
        return results, detection_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()