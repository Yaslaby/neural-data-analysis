"""
Signal Processing Module for Neuronal Data Analysis
Handles filtering, spectral analysis, and ripple detection
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, spectrogram
from PyQt5.QtCore import QThread, pyqtSignal
import warnings
warnings.filterwarnings('ignore')

class FilterProcessor:
    """Class for applying various filters to neural signals"""
    
    @staticmethod
    def bandpass_filter(data, low_freq, high_freq, fs, order=4):
        """Apply bandpass filter to data"""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if low <= 0 or high >= 1:
            raise ValueError("Filter frequencies must be within valid range")
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def lowpass_filter(data, cutoff_freq, fs, order=4):
        """Apply lowpass filter to data"""
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff >= 1:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency")
        
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data)
    
    @staticmethod
    def highpass_filter(data, cutoff_freq, fs, order=4):
        """Apply highpass filter to data"""
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff >= 1:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency")
        
        b, a = butter(order, normal_cutoff, btype='high')
        return filtfilt(b, a, data)
    
    @staticmethod
    def notch_filter(data, notch_freq, fs, quality=30):
        """Apply notch filter (e.g., for 50/60 Hz line noise)"""
        b, a = signal.iirnotch(notch_freq, quality, fs)
        return filtfilt(b, a, data)
    
    @staticmethod
    def apply_filter(data, filter_type, params, fs):
        """Apply filter based on type and parameters"""
        if filter_type == 'bandpass':
            return FilterProcessor.bandpass_filter(
                data, params['low_freq'], params['high_freq'], fs, params.get('order', 4)
            )
        elif filter_type == 'lowpass':
            return FilterProcessor.lowpass_filter(
                data, params['high_freq'], fs, params.get('order', 4)
            )
        elif filter_type == 'highpass':
            return FilterProcessor.highpass_filter(
                data, params['low_freq'], fs, params.get('order', 4)
            )
        elif filter_type == 'notch':
            return FilterProcessor.notch_filter(
                data, params.get('notch_freq', 50), fs, params.get('quality', 30)
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

class SpectralAnalyzer:
    """Class for spectral analysis operations"""
    
    @staticmethod
    def compute_psd(data, fs, nperseg=1024, method='welch'):
        """Compute power spectral density"""
        if method == 'welch':
            frequencies, psd = signal.welch(data, fs, nperseg=nperseg)
        elif method == 'periodogram':
            frequencies, psd = signal.periodogram(data, fs)
        else:
            raise ValueError(f"Unknown PSD method: {method}")
        
        return frequencies, psd
    
    @staticmethod
    def compute_spectrogram(data, fs, nperseg=256, noverlap=None):
        """Compute spectrogram"""
        if noverlap is None:
            noverlap = nperseg // 2
        
        frequencies, times, Sxx = spectrogram(
            data, fs, nperseg=nperseg, noverlap=noverlap, scaling='density'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        return frequencies, times, Sxx_db
    
    @staticmethod
    def frequency_band_power(data, fs, bands):
        """Calculate power in specific frequency bands"""
        frequencies, psd = SpectralAnalyzer.compute_psd(data, fs)
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], frequencies[band_mask])
                band_powers[band_name] = band_power
            else:
                band_powers[band_name] = 0.0
        
        return band_powers

class RippleDetector:
    """Class for detecting ripple events in neural signals"""
    
    def __init__(self, fs=1000, low_freq=80, high_freq=250, threshold=3.0, min_duration=0.015):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.threshold = threshold
        self.min_duration = min_duration
    
    def detect_ripples(self, data):
        """Detect ripple events in the signal"""
        # Filter signal in ripple band
        filtered_signal = FilterProcessor.bandpass_filter(
            data, self.low_freq, self.high_freq, self.fs
        )
        
        # Compute envelope using Hilbert transform
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope
        window_size = int(self.fs * 0.01)  # 10ms window
        envelope_smooth = np.convolve(
            envelope, np.ones(window_size)/window_size, mode='same'
        )
        
        # Calculate threshold
        mean_env = np.mean(envelope_smooth)
        std_env = np.std(envelope_smooth)
        ripple_threshold = mean_env + self.threshold * std_env
        
        # Find ripple events
        ripple_events = self._find_events(envelope_smooth, ripple_threshold)
        
        return {
            'filtered_signal': filtered_signal,
            'envelope': envelope_smooth,
            'ripple_events': ripple_events,
            'threshold': ripple_threshold,
            'detection_params': {
                'low_freq': self.low_freq,
                'high_freq': self.high_freq,
                'threshold': self.threshold,
                'min_duration': self.min_duration
            }
        }
    
    def _find_events(self, envelope, threshold):
        """Find ripple events based on threshold crossing"""
        above_threshold = envelope > threshold
        ripple_events = []
        
        in_ripple = False
        start_idx = 0
        min_samples = int(self.fs * self.min_duration)
        
        for i, above in enumerate(above_threshold):
            if above and not in_ripple:
                in_ripple = True
                start_idx = i
            elif not above and in_ripple:
                in_ripple = False
                if i - start_idx >= min_samples:
                    ripple_events.append({
                        'start': start_idx,
                        'end': i,
                        'duration': (i - start_idx) / self.fs,
                        'peak_amplitude': np.max(envelope[start_idx:i]),
                        'time': start_idx / self.fs
                    })
        
        return ripple_events

class ERDSAnalyzer:
    """Class for Event-Related Desynchronization/Synchronization analysis"""
    
    def __init__(self, fs=1000, freq_range=(2, 50), freq_step=2, 
                 window_duration=1.0, overlap=0.5, bandwidth=4.0):
        self.fs = fs
        self.freq_range = freq_range
        self.freq_step = freq_step
        self.window_duration = window_duration
        self.overlap = overlap
        self.bandwidth = bandwidth
    
    def compute_erds(self, data, baseline_start=0, baseline_end=0.5):
        """Compute ERDS time-frequency representation"""
        freqs = np.arange(self.freq_range[0], self.freq_range[1], self.freq_step)
        
        window_size = int(self.fs * self.window_duration)
        overlap_samples = int(window_size * self.overlap)
        
        n_windows = (len(data) - window_size) // (window_size - overlap_samples) + 1
        times = np.arange(n_windows) * (window_size - overlap_samples) / self.fs
        
        tf_data = np.zeros((len(freqs), n_windows))
        
        # Compute time-frequency representation
        for i, freq in enumerate(freqs):
            # Bandpass filter around frequency
            bw = self.bandwidth
            low_freq = max(freq - bw/2, 1)
            high_freq = min(freq + bw/2, self.fs/2 - 1)
            
            try:
                filtered = FilterProcessor.bandpass_filter(
                    data, low_freq, high_freq, self.fs, order=3
                )
                
                # Compute power in sliding windows
                for j in range(n_windows):
                    start_idx = j * (window_size - overlap_samples)
                    end_idx = start_idx + window_size
                    window_data = filtered[start_idx:end_idx]
                    tf_data[i, j] = np.mean(window_data**2)
            except:
                tf_data[i, :] = 0
        
        # Convert to ERDS (percentage change from baseline)
        baseline_mask = (times >= baseline_start) & (times <= baseline_end)
        if np.any(baseline_mask):
            baseline_power = np.mean(tf_data[:, baseline_mask], axis=1, keepdims=True)
            # Avoid division by zero
            baseline_power[baseline_power == 0] = 1e-12
            erds_data = ((tf_data - baseline_power) / baseline_power) * 100
        else:
            # No baseline correction
            erds_data = tf_data
        
        return {
            'erds_data': erds_data,
            'times': times,
            'freqs': freqs,
            'baseline_power': baseline_power if 'baseline_power' in locals() else None
        }

class SignalProcessingWorker(QThread):
    """Worker thread for signal processing tasks"""
    
    progress_updated = pyqtSignal(int)
    processing_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data, fs, processing_type, params):
        super().__init__()
        self.data = data
        self.fs = fs
        self.processing_type = processing_type
        self.params = params
    
    def run(self):
        """Run the signal processing task"""
        try:
            if self.processing_type == 'filter':
                result = self._apply_filter()
            elif self.processing_type == 'ripple_detection':
                result = self._detect_ripples()
            elif self.processing_type == 'erds_analysis':
                result = self._compute_erds()
            elif self.processing_type == 'spectral_analysis':
                result = self._spectral_analysis()
            else:
                raise ValueError(f"Unknown processing type: {self.processing_type}")
            
            self.processing_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _apply_filter(self):
        """Apply filter to data"""
        filter_type = self.params.get('type', 'bandpass')
        
        # Apply filter to all channels
        if self.data.ndim == 1:
            # Single channel
            filtered_data = FilterProcessor.apply_filter(
                self.data, filter_type, self.params, self.fs
            )
        else:
            # Multiple channels
            filtered_data = np.zeros_like(self.data)
            for ch in range(self.data.shape[1]):
                self.progress_updated.emit(int(ch / self.data.shape[1] * 100))
                filtered_data[:, ch] = FilterProcessor.apply_filter(
                    self.data[:, ch], filter_type, self.params, self.fs
                )
        
        self.progress_updated.emit(100)
        
        return {
            'filtered_data': filtered_data,
            'filter_params': self.params
        }
    
    def _detect_ripples(self):
        """Detect ripples in the signal"""
        detector = RippleDetector(
            fs=self.fs,
            low_freq=self.params.get('low_freq', 80),
            high_freq=self.params.get('high_freq', 250),
            threshold=self.params.get('threshold', 3.0),
            min_duration=self.params.get('min_duration', 0.015)
        )
        
        self.progress_updated.emit(50)
        result = detector.detect_ripples(self.data)
        self.progress_updated.emit(100)
        
        return result
    
    def _compute_erds(self):
        """Compute ERDS analysis"""
        analyzer = ERDSAnalyzer(
            fs=self.fs,
            freq_range=(self.params.get('freq_min', 2), self.params.get('freq_max', 50)),
            freq_step=self.params.get('freq_step', 2),
            window_duration=self.params.get('window_duration', 1.0),
            overlap=self.params.get('overlap', 0.5),
            bandwidth=self.params.get('bandwidth', 4.0)
        )
        
        result = analyzer.compute_erds(
            self.data,
            baseline_start=self.params.get('baseline_start', 0),
            baseline_end=self.params.get('baseline_end', 0.5)
        )
        
        self.progress_updated.emit(100)
        return result
    
    def _spectral_analysis(self):
        """Perform spectral analysis"""
        # Compute PSD
        frequencies, psd = SpectralAnalyzer.compute_psd(
            self.data, self.fs, nperseg=self.params.get('nperseg', 1024)
        )
        
        self.progress_updated.emit(30)
        
        # Compute spectrogram
        f_spec, t_spec, Sxx_db = SpectralAnalyzer.compute_spectrogram(
            self.data, self.fs, nperseg=self.params.get('nperseg', 256)
        )
        
        self.progress_updated.emit(60)
        
        # Compute frequency band powers
        standard_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'ripple': (80, 250)
        }
        
        band_powers = SpectralAnalyzer.frequency_band_power(
            self.data, self.fs, standard_bands
        )
        
        self.progress_updated.emit(100)
        
        return {
            'frequencies': frequencies,
            'psd': psd,
            'spectrogram_freqs': f_spec,
            'spectrogram_times': t_spec,
            'spectrogram_data': Sxx_db,
            'band_powers': band_powers
        }

# Utility functions for common signal processing tasks
def preprocess_signal(data, fs, remove_dc=True, detrend=True):
    """Basic signal preprocessing"""
    processed_data = data.copy()
    
    if remove_dc:
        # Remove DC component
        processed_data = processed_data - np.mean(processed_data, axis=0)
    
    if detrend:
        # Remove linear trend
        if processed_data.ndim == 1:
            processed_data = signal.detrend(processed_data)
        else:
            for ch in range(processed_data.shape[1]):
                processed_data[:, ch] = signal.detrend(processed_data[:, ch])
    
    return processed_data

def detect_artifacts(data, fs, threshold_std=5.0, window_size=1.0):
    """Simple artifact detection based on amplitude threshold"""
    window_samples = int(window_size * fs)
    artifacts = []
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        std_val = np.std(channel_data)
        threshold = threshold_std * std_val
        
        # Find samples exceeding threshold
        artifact_samples = np.where(np.abs(channel_data) > threshold)[0]
        
        if len(artifact_samples) > 0:
            # Group nearby artifacts
            artifact_periods = []
            start = artifact_samples[0]
            
            for i in range(1, len(artifact_samples)):
                if artifact_samples[i] - artifact_samples[i-1] > window_samples:
                    # End of current artifact period
                    artifact_periods.append((start/fs, artifact_samples[i-1]/fs, ch))
                    start = artifact_samples[i]
            
            # Add final period
            artifact_periods.append((start/fs, artifact_samples[-1]/fs, ch))
            artifacts.extend(artifact_periods)
    
    return artifacts

def calculate_signal_quality_metrics(data, fs):
    """Calculate various signal quality metrics"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    metrics = {}
    
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        
        # Basic statistics
        metrics[f'ch_{ch}_mean'] = np.mean(channel_data)
        metrics[f'ch_{ch}_std'] = np.std(channel_data)
        metrics[f'ch_{ch}_rms'] = np.sqrt(np.mean(channel_data**2))
        
        # Signal-to-noise ratio estimate
        # (using high-frequency content as noise estimate)
        high_freq_noise = FilterProcessor.highpass_filter(channel_data, 100, fs)
        signal_power = np.var(channel_data)
        noise_power = np.var(high_freq_noise)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
        metrics[f'ch_{ch}_snr'] = snr
        
        # Frequency domain metrics
        frequencies, psd = SpectralAnalyzer.compute_psd(channel_data, fs)
        peak_freq_idx = np.argmax(psd)
        metrics[f'ch_{ch}_peak_freq'] = frequencies[peak_freq_idx]
        metrics[f'ch_{ch}_total_power'] = np.trapz(psd, frequencies)
    
    return metrics