import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_data(filename):
    """
    Load neural data from file
    Currently supports .continuous files from Open Ephys
    """
    # This is where the file import happens
    data_path = Path(filename)
    if not data_path.exists():
        print(f"Error: File {filename} not found")
        return None
    
    try:
        # THIS IS THE IMPORT LINE - reads binary data from .continuous file
        data = np.fromfile(filename, dtype=np.int16)
        # Convert to microvolts
        data = data.astype(np.float64) * 1e-6
        print(f"Successfully loaded {len(data)} samples from {filename}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def calculate_noise_floor(signal, fs=20000, freq_range=(300, 500)):
    """
    Estimate noise floor using high frequency content
    Most neural signals don't have much power above 300Hz
    """
    # Get power spectral density
    freqs, psd = plt.psd(signal, Fs=fs, NFFT=4096)[:2]
    plt.close()  # Don't show the plot
    
    # Find noise frequency range
    noise_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    noise_power = np.mean(psd[noise_mask])
    
    return noise_power, freqs, psd

def calculate_signal_power(signal, fs=20000, freq_range=(1, 100)):
    """
    Calculate signal power in the frequency range of interest
    Default: 1-100Hz covers most neural activity
    """
    freqs, psd = plt.psd(signal, Fs=fs, NFFT=4096)[:2]
    plt.close()
    
    # Signal frequency range
    signal_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    signal_power = np.mean(psd[signal_mask])
    
    return signal_power

def calculate_snr_methods(signal, fs=20000):
    """
    Calculate SNR using different methods
    Returns dictionary with various SNR estimates
    """
    results = {}
    
    # Method 1: Frequency domain approach
    signal_power = calculate_signal_power(signal, fs, freq_range=(1, 100))
    noise_power, freqs, psd = calculate_noise_floor(signal, fs, freq_range=(300, 500))
    
    results['freq_domain_snr'] = 10 * np.log10(signal_power / noise_power)
    
    # Method 2: Time domain RMS approach
    # Assuming signal is the RMS of the data
    signal_rms = np.sqrt(np.mean(signal**2))
    
    # Estimate noise as standard deviation of high-pass filtered signal
    # High-pass at 300Hz to isolate noise
    from scipy.signal import butter, filtfilt
    nyquist = fs / 2
    high_cutoff = 300 / nyquist
    b, a = butter(4, high_cutoff, btype='high')
    noise_signal = filtfilt(b, a, signal)
    noise_rms = np.sqrt(np.mean(noise_signal**2))
    
    results['time_domain_snr'] = 20 * np.log10(signal_rms / noise_rms)
    
    # Method 3: Peak-to-peak approach
    signal_pp = np.max(signal) - np.min(signal)
    noise_pp = np.max(noise_signal) - np.min(noise_signal)
    results['peak_to_peak_snr'] = 20 * np.log10(signal_pp / noise_pp)
    
    # Store additional info
    results['signal_power'] = signal_power
    results['noise_power'] = noise_power
    results['freqs'] = freqs
    results['psd'] = psd
    
    return results

def calculate_snr_after_mne_processing(filename):
    """
    Calculate SNR after applying MNE processing pipeline
    This applies notch filter, bandpass filter, downsampling, etc.
    """
    # Load raw data
    signal = load_data(filename)
    if signal is None:
        return None
    
    print("Applying MNE processing pipeline...")
    
    # Apply YOUR MNE processing
    try:
        from integrated_mne_processing import process_for_ripples_mne_standard
        results = process_for_ripples_mne_standard(signal, show_plot=False)
        
        # Calculate SNR on the FILTERED data
        filtered_signal = results['ripple_filtered']
        snr_results = calculate_snr_methods(filtered_signal, fs=1000)  # Note: 1kHz after processing
        
        # Add processing info
        snr_results['processing_applied'] = True
        snr_results['original_fs'] = 20000
        snr_results['processed_fs'] = 1000
        snr_results['filters_applied'] = ['downsample', 'notch_50Hz', 'bandpass_100-250Hz']
        
        print("MNE processing completed successfully")
        return snr_results
        
    except ImportError:
        print("Error: Could not import integrated_mne_processing module")
        print("Make sure integrated_mne_processing.py is in the same directory")
        return None
    except Exception as e:
        print(f"Error during MNE processing: {e}")
        return None

def plot_snr_analysis(signal, results, fs=20000, save_plot=False, filename=None, processed=False):
    """
    Create plots showing SNR analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Adjust title based on processing
    main_title = 'SNR Analysis - MNE Processed Data' if processed else 'SNR Analysis - Raw Data'
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    # Plot 1: Time series
    time = np.arange(len(signal)) / fs
    plot_duration = min(2, len(signal)/fs)  # Show up to 2 seconds
    samples_to_plot = int(plot_duration * fs)
    
    axes[0,0].plot(time[:samples_to_plot], signal[:samples_to_plot] * 1e6)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude (µV)')
    title_suffix = f' @ {fs}Hz' if processed else f' @ {fs}Hz (Raw)'
    axes[0,0].set_title(f'Signal (first {plot_duration}s){title_suffix}')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Power Spectral Density
    freqs = results['freqs']
    psd = results['psd']
    axes[0,1].semilogy(freqs, psd)
    
    if processed:
        # For processed data, highlight the ripple band
        axes[0,1].axvspan(100, 250, alpha=0.3, color='purple', label='Ripple Band (100-250Hz)')
        axes[0,1].axvspan(300, 400, alpha=0.3, color='red', label='Noise Estimate')
        xlim_max = 500
    else:
        # For raw data, show traditional signal/noise bands
        axes[0,1].axvspan(1, 100, alpha=0.3, color='green', label='Signal (1-100Hz)')
        axes[0,1].axvspan(300, 500, alpha=0.3, color='red', label='Noise (300-500Hz)')
        xlim_max = 1000
    
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Power Spectral Density')
    axes[0,1].set_title('Frequency Domain Analysis')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(0, xlim_max)
    
    # Plot 3: SNR comparison
    snr_methods = ['freq_domain_snr', 'time_domain_snr', 'peak_to_peak_snr']
    snr_values = [results[method] for method in snr_methods]
    method_names = ['Frequency\nDomain', 'Time Domain\n(RMS)', 'Peak-to-Peak']
    
    bars = axes[1,0].bar(method_names, snr_values, color=['blue', 'orange', 'green'])
    axes[1,0].set_ylabel('SNR (dB)')
    axes[1,0].set_title('SNR by Different Methods')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, snr_values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 4: Signal quality assessment
    y_pos = 0.9
    axes[1,1].text(0.1, y_pos, f'Signal Quality Report:', fontsize=14, fontweight='bold')
    y_pos -= 0.1
    
    if processed:
        axes[1,1].text(0.1, y_pos, f'Processing: MNE Pipeline Applied', fontsize=10, color='blue')
        y_pos -= 0.08
        axes[1,1].text(0.1, y_pos, f'Filters: {", ".join(results.get("filters_applied", []))}', fontsize=9)
        y_pos -= 0.08
    
    axes[1,1].text(0.1, y_pos, f'Frequency Domain SNR: {results["freq_domain_snr"]:.1f} dB')
    y_pos -= 0.08
    axes[1,1].text(0.1, y_pos, f'Time Domain SNR: {results["time_domain_snr"]:.1f} dB')
    y_pos -= 0.08
    axes[1,1].text(0.1, y_pos, f'Peak-to-Peak SNR: {results["peak_to_peak_snr"]:.1f} dB')
    y_pos -= 0.1
    
    # Quality assessment
    avg_snr = np.mean(snr_values)
    if avg_snr > 20:
        quality = "Excellent"
        color = "green"
    elif avg_snr > 15:
        quality = "Good"
        color = "blue"
    elif avg_snr > 10:
        quality = "Fair"
        color = "orange"
    else:
        quality = "Poor"
        color = "red"
    
    axes[1,1].text(0.1, y_pos, f'Overall Quality: {quality}', fontsize=12, 
                   fontweight='bold', color=color)
    y_pos -= 0.08
    axes[1,1].text(0.1, y_pos, f'Average SNR: {avg_snr:.1f} dB', fontweight='bold')
    
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    if save_plot and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Calculate Signal-to-Noise Ratio for neural data')
    parser.add_argument('filename', help='Path to data file (.continuous)')
    parser.add_argument('--fs', type=int, default=20000, help='Sampling frequency (default: 20000)')
    parser.add_argument('--plot', action='store_true', help='Show analysis plots')
    parser.add_argument('--save', type=str, help='Save plot to file')
    parser.add_argument('--processed', action='store_true', help='Apply MNE processing pipeline before SNR calculation')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.filename).exists():
        print(f"Error: File {args.filename} not found")
        return
    
    print(f"Loading data from {args.filename}...")
    
    if args.processed:
        # Calculate SNR after MNE processing
        print("Using MNE processed data for SNR calculation...")
        results = calculate_snr_after_mne_processing(args.filename)
        if results is None:
            return
        
        # Load the processed signal for plotting
        signal = load_data(args.filename)
        from integrated_mne_processing import process_for_ripples_mne_standard
        mne_results = process_for_ripples_mne_standard(signal, show_plot=False)
        signal = mne_results['ripple_filtered']
        fs = 1000
        
    else:
        # Calculate SNR on raw data
        signal = load_data(args.filename)
        if signal is None:
            return
        
        print(f"Data loaded: {len(signal)} samples, {len(signal)/args.fs:.1f} seconds")
        print("Calculating SNR on raw data...")
        results = calculate_snr_methods(signal, args.fs)
        fs = args.fs
    
    # Print results
    print("\n" + "="*50)
    if args.processed:
        print("SNR ANALYSIS - MNE PROCESSED DATA")
    else:
        print("SNR ANALYSIS - RAW DATA")
    print("="*50)
    print(f"Frequency Domain SNR: {results['freq_domain_snr']:.2f} dB")
    print(f"Time Domain SNR:      {results['time_domain_snr']:.2f} dB") 
    print(f"Peak-to-Peak SNR:     {results['peak_to_peak_snr']:.2f} dB")
    
    avg_snr = np.mean([results['freq_domain_snr'], 
                       results['time_domain_snr'], 
                       results['peak_to_peak_snr']])
    print(f"\nAverage SNR: {avg_snr:.2f} dB")
    
    if args.processed:
        print(f"Processing applied: {', '.join(results.get('filters_applied', []))}")
        print(f"Sampling rate: {results.get('original_fs', 20000)} Hz → {results.get('processed_fs', 1000)} Hz")
    
    # Quality assessment
    if avg_snr > 20:
        print("Signal Quality: EXCELLENT - Great for analysis")
    elif avg_snr > 15:
        print("Signal Quality: GOOD - Suitable for most analyses")
    elif avg_snr > 10:
        print("Signal Quality: FAIR - May need preprocessing")
    else:
        print("Signal Quality: POOR - Consider noise reduction")
    
    # Show plots if requested
    if args.plot or args.save:
        plot_snr_analysis(signal, results, fs, args.save is not None, args.save, args.processed)

if __name__ == "__main__":
    # If no command line args, run with default file
    import sys
    if len(sys.argv) == 1:
        # Default file path - change this to your data file
        default_file = "/Users/yaslaby/Documents/PyGt5_project/Channels/120_CH1.continuous"
        if Path(default_file).exists():
            print("Choose analysis type:")
            print("1. Raw data SNR")
            print("2. MNE processed data SNR") 
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "2":
                print(f"Running MNE processed SNR analysis on: {default_file}")
                sys.argv.extend([default_file, "--processed", "--plot"])
            else:
                print(f"Running raw data SNR analysis on: {default_file}")
                sys.argv.extend([default_file, "--plot"])
        else:
            print("Usage: python snr_calculator.py <datafile.continuous> [--processed] [--plot] [--save output.png]")
            print("Or modify the default_file path in the script")
            sys.exit(1)
    
    main()