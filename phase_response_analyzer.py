
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from pathlib import Path

def create_mne_filter(fs_orig=20000, fs_new=1000):
    """
    Recreate MNE's anti-aliasing filter parameters
    
    Parameters:
    -----------
    fs_orig : int
        Original sampling frequency (Hz)
    fs_new : int  
        Target sampling frequency (Hz)
        
    Returns:
    --------
    h : array
        Filter coefficients
    w : array
        Frequency points
    H : array
        Complex frequency response
    filter_params : dict
        Filter parameters used
    """
    
    # MNE's filter design parameters
    nyquist = fs_new / 2  # New Nyquist frequency
    h_freq = nyquist  # Cutoff frequency
    
    # MNE's transition bandwidth formula
    transition_width = min(max(h_freq * 0.25, 2.0), fs_orig / 2.0 - h_freq)
    
    # MNE's filter length calculation (3.3 factor for Hamming window with firwin)
    filter_len = int(3.3 * fs_orig / transition_width)
    if filter_len % 2 == 0:
        filter_len += 1
    
    # Create the filter
    h = signal.firwin(filter_len, h_freq, fs=fs_orig, window='hamming')
    w, H = signal.freqz(h, worN=4096, fs=fs_orig)
    
    filter_params = {
        'fs_orig': fs_orig,
        'fs_new': fs_new,
        'nyquist': nyquist,
        'h_freq': h_freq,
        'transition_width': transition_width,
        'filter_len': filter_len,
        'window': 'hamming'
    }
    
    return h, w, H, filter_params

def analyze_phase_response(w, H, fs_orig, filter_params):
    """
    Comprehensive phase response analysis
    
    Parameters:
    -----------
    w : array
        Frequency points
    H : array
        Complex frequency response
    fs_orig : int
        Original sampling frequency
    filter_params : dict
        Filter parameters
        
    Returns:
    --------
    phase_analysis : dict
        Complete phase analysis results
    """
    
    # Calculate phase responses
    phase_response = np.angle(H)  # Phase in radians
    phase_response_deg = np.degrees(phase_response)  # Phase in degrees
    phase_unwrapped = np.unwrap(phase_response)
    phase_unwrapped_deg = np.degrees(phase_unwrapped)
    
    # Calculate group delay
    group_delay = -np.diff(phase_unwrapped) / (2 * np.pi * np.diff(w) / fs_orig)
    w_gd = (w[:-1] + w[1:]) / 2  # Midpoint frequencies for group delay
    
    # Analyze passband (well below cutoff)
    cutoff_freq = filter_params['h_freq']
    passband_mask = w <= (cutoff_freq * 0.8)  # 80% of cutoff frequency
    
    # Linear phase analysis in passband
    phase_linearity = {}
    if len(w[passband_mask]) > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            w[passband_mask], phase_unwrapped_deg[passband_mask])
        linear_phase_fit = slope * w[passband_mask] + intercept
        phase_deviation = phase_unwrapped_deg[passband_mask] - linear_phase_fit
        rms_deviation = np.sqrt(np.mean(phase_deviation**2))
        
        phase_linearity = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'rms_deviation': rms_deviation,
            'linear_fit': linear_phase_fit,
            'deviation': phase_deviation
        }
    
    # Group delay analysis
    group_delay_analysis = {}
    if len(group_delay) > 0:
        passband_gd_mask = w_gd <= (cutoff_freq * 0.8)
        if np.any(passband_gd_mask):
            avg_group_delay = np.mean(group_delay[passband_gd_mask])
            group_delay_std = np.std(group_delay[passband_gd_mask])
            
            group_delay_analysis = {
                'average': avg_group_delay,
                'std': group_delay_std,
                'avg_time_ms': avg_group_delay / fs_orig * 1000,
                'passband_mask': passband_gd_mask
            }
    
    # Key phase characteristics
    phase_at_dc = phase_unwrapped_deg[0]
    cutoff_idx = np.argmin(np.abs(w - cutoff_freq))
    phase_at_cutoff = phase_unwrapped_deg[cutoff_idx]
    
    return {
        'phase_rad': phase_response,
        'phase_deg': phase_response_deg,
        'phase_unwrapped_deg': phase_unwrapped_deg,
        'group_delay': group_delay,
        'w_gd': w_gd,
        'passband_mask': passband_mask,
        'phase_linearity': phase_linearity,
        'group_delay_analysis': group_delay_analysis,
        'phase_at_dc': phase_at_dc,
        'phase_at_cutoff': phase_at_cutoff,
        'cutoff_freq': cutoff_freq
    }

def plot_phase_analysis(w, H, phase_analysis, filter_params):
    """
    Create comprehensive phase response plots
    """
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Extract analysis results
    phase_unwrapped_deg = phase_analysis['phase_unwrapped_deg']
    group_delay = phase_analysis['group_delay']
    w_gd = phase_analysis['w_gd']
    passband_mask = phase_analysis['passband_mask']
    cutoff_freq = phase_analysis['cutoff_freq']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phase Frequency Response Analysis: MNE Anti-Aliasing Filter', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Magnitude and Phase together
    ax = axes[0, 0]
    ax1 = ax
    ax2 = ax1.twinx()
    
    # Magnitude response
    line1 = ax1.plot(w, 20*np.log10(np.abs(H)), 'b-', linewidth=2, label='Magnitude (dB)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(-80, 5)
    ax1.grid(True, alpha=0.3)
    
    # Phase response
    line2 = ax2.plot(w, phase_unwrapped_deg, 'r-', linewidth=2, label='Phase (degrees)')
    ax2.set_ylabel('Phase (degrees)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add cutoff line
    ax1.axvline(cutoff_freq, color='gray', linestyle='--', alpha=0.7, label=f'Cutoff ({cutoff_freq} Hz)')
    
    ax1.set_title('Magnitude and Phase Response')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # Plot 2: Phase response with linearity analysis
    ax = axes[0, 1]
    ax.plot(w, phase_unwrapped_deg, 'r-', linewidth=2, label='Unwrapped phase')
    ax.axvline(cutoff_freq, color='gray', linestyle='--', alpha=0.7, 
               label=f'Cutoff ({cutoff_freq} Hz)')
    
    # Add linear fit if available
    if 'r_squared' in phase_analysis['phase_linearity']:
        linear_fit = phase_analysis['phase_linearity']['linear_fit']
        r_squared = phase_analysis['phase_linearity']['r_squared']
        ax.plot(w[passband_mask], linear_fit, 'g--', linewidth=2, 
                label=f'Linear fit (R²={r_squared:.4f})')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Phase Response with Linear Fit')
    ax.set_xlim(0, 1000)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Group delay
    ax = axes[1, 0]
    if len(group_delay) > 0:
        ax.plot(w_gd, group_delay, 'purple', linewidth=2, label='Group delay')
        ax.axvline(cutoff_freq, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Cutoff ({cutoff_freq} Hz)')
        
        # Add average group delay line if available
        if 'average' in phase_analysis['group_delay_analysis']:
            avg_gd = phase_analysis['group_delay_analysis']['average']
            ax.axhline(avg_gd, color='red', linestyle=':', alpha=0.7, 
                       label=f'Average: {avg_gd:.1f} samples')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Group Delay (samples)')
        ax.set_title('Group Delay (Phase Linearity Indicator)')
        ax.set_xlim(0, 1000)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Phase linearity assessment
    ax = axes[1, 1]
    if 'deviation' in phase_analysis['phase_linearity']:
        deviation = phase_analysis['phase_linearity']['deviation']
        rms_dev = phase_analysis['phase_linearity']['rms_deviation']
        
        ax.plot(w[passband_mask], deviation, 'orange', linewidth=2, 
                label='Phase deviation from linear')
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Deviation (degrees)')
        ax.set_title('Phase Linearity in Passband')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add RMS deviation text
        ax.text(0.05, 0.95, f'RMS deviation: {rms_dev:.3f}°', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()

def print_analysis_summary(phase_analysis, filter_params):
    """
    Print comprehensive analysis summary
    """
    
    print("\n" + "="*60)
    print("PHASE FREQUENCY RESPONSE ANALYSIS SUMMARY")
    print("="*60)
    
    # Filter parameters
    print("Filter Parameters:")
    print("-" * 20)
    print(f"Original sampling rate: {filter_params['fs_orig']} Hz")
    print(f"Target sampling rate: {filter_params['fs_new']} Hz")
    print(f"Cutoff frequency: {filter_params['h_freq']} Hz")
    print(f"Transition bandwidth: {filter_params['transition_width']:.1f} Hz")
    print(f"Filter length: {filter_params['filter_len']} samples")
    print(f"Window type: {filter_params['window']}")
    
    # Phase characteristics
    print(f"\nPhase Response Analysis:")
    print("-" * 25)
    print(f"Phase at DC (0 Hz): {phase_analysis['phase_at_dc']:.3f} degrees")
    print(f"Phase at cutoff: {phase_analysis['phase_at_cutoff']:.3f} degrees")
    
    # Linear phase assessment
    if 'r_squared' in phase_analysis['phase_linearity']:
        linearity = phase_analysis['phase_linearity']
        print(f"Phase linearity (R²): {linearity['r_squared']:.6f}")
        print(f"Phase slope: {linearity['slope']:.3f} deg/Hz")
        print(f"RMS deviation from linear: {linearity['rms_deviation']:.4f} degrees")
    
    # Group delay assessment
    if 'average' in phase_analysis['group_delay_analysis']:
        gd_analysis = phase_analysis['group_delay_analysis']
        print(f"Average group delay: {gd_analysis['average']:.1f} samples")
        print(f"Group delay time: {gd_analysis['avg_time_ms']:.3f} ms")
        print(f"Group delay variation: ±{gd_analysis['std']:.2f} samples")
    
    # Filter classification
    print(f"\nFilter Type Classification:")
    print("-" * 28)
    
    phase_at_dc = abs(phase_analysis['phase_at_dc'])
    
    if phase_at_dc < 0.1:  # Within 0.1 degrees of zero
        if 'std' in phase_analysis['group_delay_analysis']:
            gd_std = phase_analysis['group_delay_analysis']['std']
            if gd_std < 1.0:  # Very low group delay variation
                print("✓ ZERO-PHASE FILTER (Excellent)")
                print("  • No phase distortion")
                print("  • Constant group delay")
                print("  • Perfect for offline neural signal processing")
            else:
                print("✓ NEAR ZERO-PHASE FILTER (Very Good)")
                print("  • Minimal phase distortion")
        else:
            print("✓ ZERO-PHASE FILTER (Excellent)")
    elif 'r_squared' in phase_analysis['phase_linearity'] and phase_analysis['phase_linearity']['r_squared'] > 0.9999:
        slope = phase_analysis['phase_linearity']['slope']
        time_delay_ms = -slope / (360 * filter_params['fs_orig']) * 1000
        print("✓ LINEAR-PHASE FILTER (Good)")
        print(f"  • Constant time delay: {time_delay_ms:.3f} ms")
        print("  • Preserves waveform shapes")
    else:
        print("⚠ NON-LINEAR PHASE FILTER (Caution)")
        print("  • May introduce phase distortion")
        print("  • Could affect waveform shapes")
    
    # Suitability assessment
    print(f"\nSuitability for Neural Signal Processing:")
    print("-" * 40)
    print("✓ MNE uses zero-phase filtering by default")
    print("✓ No time delay introduced to signals")
    print("✓ Preserves LFP waveform morphology")
    print("✓ Maintains phase relationships between channels")
    print("✓ Excellent for ripple detection and timing analysis")
    print("✓ Suitable for cross-frequency coupling studies")
    
    if phase_at_dc < 0.1:
        print("\n🎯 RECOMMENDATION: Filter is EXCELLENT for your LFP analysis")
    elif 'r_squared' in phase_analysis['phase_linearity'] and phase_analysis['phase_linearity']['r_squared'] > 0.999:
        print("\n✅ RECOMMENDATION: Filter is GOOD for your LFP analysis")
    else:
        print("\n⚠️  RECOMMENDATION: Consider filter optimization for critical timing analyses")

def main():
    """
    Main function to run complete phase response analysis
    """
    
    print("Phase Frequency Response Analyzer for MNE Anti-Aliasing Filter")
    print("=" * 65)
    
    # Create filter (matching your original parameters)
    fs_orig = 20000  # Original sampling rate
    fs_new = 1000    # Target sampling rate
    
    print(f"Analyzing MNE filter: {fs_orig} Hz → {fs_new} Hz")
    
    # Generate filter
    h, w, H, filter_params = create_mne_filter(fs_orig, fs_new)
    
    # Analyze phase response
    phase_analysis = analyze_phase_response(w, H, fs_orig, filter_params)
    
    # Create plots
    plot_phase_analysis(w, H, phase_analysis, filter_params)
    
    # Print summary
    print_analysis_summary(phase_analysis, filter_params)

if __name__ == "__main__":
    main()