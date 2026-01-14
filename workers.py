"""
Worker classes for background processing operations
"""
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QThread
from PyQt5.QtWidgets import QApplication
import numpy as np


class DownsampleWorker(QObject):
    """Worker for downsampling operations"""
    progress_updated = pyqtSignal(int, str)
    processing_completed = pyqtSignal(object, object, object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.target_fs = None
        self.original_fs = None
        self.original_timestamps = None
    
    def set_data(self, data, target_fs, original_fs, original_timestamps):
        self.data = data
        self.target_fs = target_fs
        self.original_fs = original_fs
        self.original_timestamps = original_timestamps
    
    def run(self):
        try:
            import mne
            
            self.progress_updated.emit(20, "Creating MNE object...")
            
            if self.data.ndim == 1:
                n_channels = 1
                data_for_mne = self.data.reshape(1, -1)
                info = mne.create_info(['CH1'], self.original_fs, ['seeg'])
            else:
                n_channels = self.data.shape[1]
                data_for_mne = self.data.T
                ch_names = [f'CH{i+1}' for i in range(n_channels)]
                info = mne.create_info(ch_names, self.original_fs, ['seeg'] * n_channels)
            
            raw = mne.io.RawArray(data_for_mne, info, verbose=False)
            
            self.progress_updated.emit(60, "Downsampling...")
            raw_downsampled = raw.copy().resample(self.target_fs, verbose=False)
            downsampled_data = raw_downsampled.get_data().T
            
            self.progress_updated.emit(90, "Creating timestamps...")
            
            if self.original_timestamps is not None:
                start_time = self.original_timestamps[0]
                end_time = self.original_timestamps[-1]
                new_timestamps = np.linspace(start_time, end_time, len(downsampled_data))
            else:
                new_timestamps = np.arange(len(downsampled_data)) / self.target_fs
            
            self.progress_updated.emit(100, "Complete!")
            
            new_header = {'sampleRate': self.target_fs}
            self.processing_completed.emit(downsampled_data, new_timestamps, new_header)
            
        except Exception as e:
            self.error_occurred.emit(f"Downsampling failed: {str(e)}")


class PreprocessingWorker(QObject):
    """Worker for preprocessing operations with comparison output"""
    
    progress_updated = pyqtSignal(int, str)
    processing_completed = pyqtSignal(object, object, object, object, list, int, int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.header = None
        self.timestamps = None
        self.params = None
        self.is_cancelled = False
        self.mutex = QMutex()
    
    def set_data(self, data, header, timestamps, params):
        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.params = params
        self.is_cancelled = False
    
    def cancel(self):
        self.mutex.lock()
        self.is_cancelled = True
        self.mutex.unlock()
    
    def run(self):
        try:
            if self.is_cancelled:
                return
            
            channels = self.params['channels'][:4]
            
            self.progress_updated.emit(5, "Initializing preprocessing...")
            QApplication.processEvents()
            
            if self.data.ndim > 1:
                if len(channels) == 1:
                    raw_data = self.data[:, channels[0]].reshape(-1, 1)
                else:
                    raw_data = self.data[:, channels]
            else:
                raw_data = self.data.reshape(-1, 1)
            
            raw_timestamps = self.timestamps[:len(raw_data)]
            original_fs = self.params['original_fs']
            target_fs = self.params['target_fs']
            
            self.progress_updated.emit(10, f"Processing {len(channels)} channels...")
            QApplication.processEvents()
            
            processed_channels = []
            total_channels = raw_data.shape[1]
            
            for i in range(total_channels):
                if self.is_cancelled:
                    return
                
                progress = 10 + (i * 70 // total_channels)
                self.progress_updated.emit(progress, f"Processing channel {i+1}/{total_channels}...")
                QApplication.processEvents()
                ch_data = raw_data[:, i]
                
                try:
                    processed_data = self._process_with_mne(ch_data, original_fs, target_fs, i)
                    processed_channels.append(processed_data)
                    QApplication.processEvents()
                except Exception as e:
                    self.error_occurred.emit(f"Error processing channel {i+1}: {str(e)}")
                    return
                
                QThread.msleep(50)
                QApplication.processEvents()
            
            if self.is_cancelled:
                return
            
            self.progress_updated.emit(85, "Finalizing processed data...")
            
            if len(processed_channels) > 1:
                proc_data = np.column_stack(processed_channels)
            else:
                proc_data = processed_channels[0].reshape(-1, 1)
            
            if self.timestamps is not None:
                start_time = self.timestamps[0]
                end_time = self.timestamps[-1]
                proc_timestamps = np.linspace(start_time, end_time, len(proc_data))
            else:
                proc_timestamps = np.arange(len(proc_data)) / target_fs
            
            if 'channel_files' in self.header and self.header['channel_files']:
                channel_names = [self.header['channel_files'][i] for i in channels]
            else:
                channel_names = [f"CH{i+1}" for i in channels]
            
            self.progress_updated.emit(100, "Processing complete!")
            
            self.processing_completed.emit(
                raw_data, raw_timestamps, proc_data, proc_timestamps,
                channel_names, original_fs, target_fs
            )
            
        except Exception as e:
            self.error_occurred.emit(f"Processing failed: {str(e)}")
    
    def _process_with_mne(self, data, original_fs, target_fs, channel_idx):
        try:
            import mne
            from mne.filter import notch_filter
            
            info = mne.create_info([f'CH{channel_idx+1}'], original_fs, ['seeg'])
            raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
            
            if target_fs != original_fs:
                raw = raw.copy().resample(target_fs, verbose=False)
            
            if self.params['notch_enabled']:
                notch_freqs = self.params.get('notch_frequencies', [50])
                # Filter out frequencies above Nyquist
                valid_freqs = [f for f in notch_freqs if f < target_fs / 2]
                
                if valid_freqs:
                    raw._data[0] = notch_filter(
                        raw._data[0], target_fs, freqs=valid_freqs,
                        method='fir', phase='zero', verbose=False
                    )

            if self.params['bandpass_enabled']:
                raw = raw.copy().filter(
                    l_freq=self.params['low_cutoff'],
                    h_freq=self.params['high_cutoff'],
                    method='fir', phase='zero', verbose=False
                )
            
            return raw._data[0]
            
        except Exception as e:
            raise Exception(f"MNE processing failed: {str(e)}")