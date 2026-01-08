"""
Data Loader Module
Handles file loading, downsampling, and multi-channel data operations
"""
import os
import traceback
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QDialog, QProgressDialog, QMessageBox,QInputDialog
)
from PyQt5.QtCore import Qt, QThread

from MultiChannelLoader import load_single_channel
from dialogs import DownsampleDialog
from workers import DownsampleWorker


class DataLoader:
    """Mixin class containing data loading methods for OpenEphysMainWindow"""

    def load_single_file(self):
        """Load single .continuous file with immediate downsample dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select .continuous file',
            filter="Continuous files (*.continuous)"
        )
        
        if not file_path:
            return
        
        try:
            result = load_single_channel(file_path, verbose=True)
            data = result['data']
            timestamps = result['timestamps']
            header = result['header']

            # Ensure timestamps start at zero
            if timestamps is not None and len(timestamps) > 0 and timestamps[0] != 0:
                timestamps = timestamps - timestamps[0]
            
            self.show_downsample_dialog(data, header, timestamps, file_path)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")
            traceback.print_exc()
    def load_preprocessed_mat(self):
        """Load preprocessed data from .mat file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select preprocessed .mat file',
            filter="MATLAB Files (*.mat)"
        )
        
        if not file_path:
            return
        
        try:
            mat = loadmat(file_path)
            
            # Find the data array
            data = None
            fs = None
            
            # Check common variable names
            for name in ['data', 'signal', 'lfp', 'LFP', 'eeg', 'preprocessed']:
                if name in mat:
                    data = mat[name].squeeze()
                    break
            
            for name in ['fs', 'Fs', 'sampleRate', 'sample_rate', 'sr']:
                if name in mat:
                    fs = float(mat[name].squeeze())
                    break
            
            # If not found, grab the largest array
            if data is None:
                for key, val in mat.items():
                    if not key.startswith('_') and isinstance(val, np.ndarray) and val.size > 1000:
                        data = val.squeeze()
                        break
            
            if data is None:
                available = [k for k in mat.keys() if not k.startswith('_')]
                QMessageBox.warning(self, "Data Not Found",
                    f"Could not find signal data.\nAvailable variables: {', '.join(available)}")
                return
            
            # Ask for sampling rate if not found
            if fs is None:
                fs, ok = QInputDialog.getDouble(self, "Sampling Rate",
                    "Enter sampling rate (Hz):", 1000, 1, 100000, 0)
                if not ok:
                    return
            
            # Ensure shape is (samples, channels)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.shape[0] < data.shape[1]:
                data = data.T
            
            timestamps = np.arange(len(data)) / fs
            
            header = {
                'sampleRate': fs,
                'preprocessed': True,  # This flag skips downsampling & notch
                'channel_count': data.shape[1],
                'channel_files': [f"CH{i+1}" for i in range(data.shape[1])]
            }
            
            name = os.path.basename(file_path).replace('.mat', '') + ' (preprocessed)'
            
            dataset = {
                "name": name,
                "data": data.astype(np.float64),
                "header": header,
                "file_path": file_path,
                "timestamps": timestamps
            }
            
            self.add_dataset(dataset)
            
            QMessageBox.information(self, "Loaded",
                f"Preprocessed data loaded!\n\n"
                f"Channels: {data.shape[1]}\n"
                f"Samples: {len(data):,}\n"
                f"Rate: {fs:.0f} Hz\n"
                f"Duration: {timestamps[-1]:.1f}s\n\n"
                f"Notch filter will be skipped.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")
            traceback.print_exc()
            
    def show_downsample_dialog(self, data, header, timestamps, file_path):
        """Show downsample dialog and handle downsampling"""
        dialog = DownsampleDialog(data, header, timestamps, file_path, self)
        
        if dialog.exec_() == QDialog.Accepted:
            target_fs = dialog.get_target_frequency()
            if target_fs is None:
                return
                
            original_fs = header.get('sampleRate', 20000)
            self.start_downsampling(data, target_fs, original_fs, timestamps, file_path)

    def start_downsampling(self, data, target_fs, original_fs, timestamps, file_path):
        """Start downsampling with proper cleanup check"""
        try:
            self.cleanup_worker_thread()
            
            if self.worker_thread is not None:
                return
            
            self.progress_dialog = QProgressDialog("Downsampling data...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()
            
            self.worker = DownsampleWorker()
            self.worker.set_data(data, target_fs, original_fs, timestamps)
            
            self.worker_thread = QThread()
            self.worker.moveToThread(self.worker_thread)
            
            # Connect signals
            self.worker_thread.started.connect(self.worker.run)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.processing_completed.connect(
                lambda d, t, h: self.on_downsample_complete(d, t, h, file_path, data.shape[1] if data.ndim > 1 else 1)
            )
            self.worker.error_occurred.connect(self.on_downsample_error)
            
            # Cleanup connections
            self.worker.processing_completed.connect(self.worker_thread.quit)
            self.worker.error_occurred.connect(self.worker_thread.quit)
            
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Downsample Error", f"Failed to start:\n{str(e)}")

    def update_progress(self, value, message):
        """Update progress dialog"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()

    def on_downsample_complete(self, data, timestamps, header, file_path, n_channels=1):
        """Handle downsample completion - preserve multichannel structure"""
        try:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            
            filename = os.path.basename(file_path)
            name = filename.replace('.continuous', '')
            
            # Preserve multichannel structure
            if n_channels > 1:
                if data.ndim == 1:
                    data = data.reshape(-1, n_channels)
                name = f"Multi-channel ({n_channels} channels)"
                
                if hasattr(self, '_multichannel_header'):
                    if 'channel_files' in self._multichannel_header:
                        header['channel_files'] = self._multichannel_header['channel_files']
                        header['channel_count'] = self._multichannel_header.get('channel_count', n_channels)
                    delattr(self, '_multichannel_header')
            elif data.ndim == 1:
                data = data.reshape(-1, 1)
            
            dataset = {
                "name": name,
                "data": data,
                "header": header,
                "file_path": file_path,
                "timestamps": timestamps
            }
            
            self.add_dataset(dataset)
            
            QMessageBox.information(self, "Ready for Analysis", 
                                f"Data ready at {header['sampleRate']:.0f}Hz\n"
                                f"Channels: {data.shape[1]}\n"
                                f"Samples: {len(data):,}\n"
                                f"Duration: {timestamps[-1] - timestamps[0]:.1f}s\n\n"
                                f"Use Edit → Preprocess to apply filters")
            
        except Exception as e:
            QMessageBox.critical(self, "Complete Error", f"Failed to complete downsampling:\n{str(e)}")

    def on_downsample_error(self, error_msg):
        """Handle downsample error"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        QMessageBox.critical(self, "Downsample Error", error_msg)

    def cancel_downsampling(self):
        """Cancel downsampling"""
        self.cleanup_worker_thread()
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

    def load_multiple_channels(self):
        """Load multiple .continuous files"""
        self.multichannel_loader.select_and_load_files(self.on_multichannel_loaded)

    def on_multichannel_loaded(self, data, timestamps, header, description):
        """Handle multi-channel data loading completion WITH downsample dialog"""
        try:
            # Ensure timestamps start at zero
            if timestamps is not None and len(timestamps) > 0 and timestamps[0] != 0:
                timestamps = timestamps - timestamps[0]
            
            # Store for use in downsample completion
            self._multichannel_header = header
            
            self.show_downsample_dialog(data, header, timestamps, description)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process multi-channel data:\n{str(e)}")
            traceback.print_exc()
