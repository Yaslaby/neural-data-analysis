# MultiChannelLoader.py
import os
import numpy as np
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QProgressDialog, 
                             QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import struct

def read_openephys_header(f):
    """Read header information from Open Ephys file"""
    header = {}
    
    # Read and process header data
    header_string = f.read(1024).decode('utf-8', errors='ignore').replace('\n', '').replace('header.', '')
    
    # Parse each key = value string
    for pair in header_string.split(';'):
        if '=' in pair:
            try:
                key, value = pair.split(' = ', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert numeric values
                if key in ['bitVolts', 'sampleRate']:
                    header[key] = float(value)
                elif key in ['blockLength', 'bufferSize', 'header_bytes']:
                    header[key] = int(value)
                else:
                    header[key] = value
            except ValueError:
                continue
    
    return header

def load_continuous_with_timestamps(filepath, verbose=False):
    """Load continuous data with absolute timestamps"""
    if verbose:
        print(f"Loading: {os.path.basename(filepath)}")
    
    # Record marker for validation
    record_marker = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)
    
    timestamps = []
    samples = []
    
    try:
        with open(filepath, 'rb') as f:
            # Read header
            header = read_openephys_header(f)
            
            # Get block length and sample rate
            block_length = header.get('blockLength', 1024)
            sample_rate = header.get('sampleRate', 30000.0)
            bit_volts = header.get('bitVolts', 0.195)
            
            # Calculate record size
            record_size = 2 * block_length + 22  # data + metadata
            
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(1024)  # Back to start of data
            
            record_count = 0
            
            while f.tell() < file_size - record_size:
                try:
                    # Read timestamp (8 bytes, little-endian)
                    timestamp_bytes = f.read(8)
                    if len(timestamp_bytes) < 8:
                        break
                    timestamp = struct.unpack('<Q', timestamp_bytes)[0]
                    
                    # Read number of samples (2 bytes, little-endian)
                    n_samples_bytes = f.read(2)
                    if len(n_samples_bytes) < 2:
                        break
                    n_samples = struct.unpack('<H', n_samples_bytes)[0]
                    
                    # Validate block length
                    if n_samples != block_length:
                        if verbose:
                            print(f"Warning: Block length mismatch at record {record_count}")
                        # Skip this record
                        f.seek(2 + n_samples * 2 + 10, 1)
                        continue
                    
                    # Read recording number (2 bytes, big-endian) - skip it
                    f.read(2)
                    
                    # Read sample data (n_samples * 2 bytes, big-endian signed integers)
                    sample_bytes = f.read(n_samples * 2)
                    if len(sample_bytes) < n_samples * 2:
                        break
                    
                    # Convert to samples
                    block_samples = np.frombuffer(sample_bytes, dtype='>i2').astype(np.float32)
                    
                    # Convert to microvolts
                    block_samples = block_samples * bit_volts
                    
                    # Read and validate record marker
                    marker_bytes = f.read(10)
                    if len(marker_bytes) < 10:
                        break
                    marker = np.frombuffer(marker_bytes, dtype=np.uint8)
                    
                    if not np.array_equal(marker, record_marker):
                        if verbose:
                            print(f"Warning: Invalid record marker at record {record_count}")
                    
                    # Store data
                    timestamps.append(timestamp)
                    samples.append(block_samples)
                    record_count += 1
                    
                except struct.error:
                    if verbose:
                        print(f"Struct error at record {record_count}")
                    break
                except Exception as e:
                    if verbose:
                        print(f"Error reading record {record_count}: {e}")
                    break
    
    except Exception as e:
        raise Exception(f"Failed to read {filepath}: {str(e)}")
    
    # Process results
    if len(samples) == 0:
        raise Exception(f"No valid data found in {filepath}")
    
    # Concatenate all samples
    data = np.concatenate(samples)
    
    # Convert timestamps to seconds (absolute time)
    timestamps_array = np.array(timestamps, dtype=np.float64)
    if len(timestamps_array) > 0:
        # Convert from sample count to absolute time in seconds
        time_per_sample = 1.0 / sample_rate
        absolute_timestamps = timestamps_array * time_per_sample
        
        # Create full timestamp array for all samples
        full_timestamps = []
        for i, (start_time, block_data) in enumerate(zip(absolute_timestamps, samples)):
            block_times = start_time + np.arange(len(block_data)) * time_per_sample
            full_timestamps.append(block_times)
        
        full_timestamps = np.concatenate(full_timestamps)
    else:
        # Fallback to relative time
        full_timestamps = np.arange(len(data)) / sample_rate
    
    result = {
        'data': data,
        'timestamps': full_timestamps,
        'header': header,
        'sample_rate': sample_rate,
        'record_count': record_count
    }
    
    if verbose:
        print(f"  Loaded {len(data)} samples, {record_count} records")
        print(f"  Time range: {full_timestamps[0]:.3f} to {full_timestamps[-1]:.3f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
    
    result = {
        'data': data,
        'timestamps': full_timestamps,
        'header': header,
        'sample_rate': sample_rate,
        'record_count': record_count
    }
    if verbose:
        print(f"\n--- Detailed Loading Stats for {os.path.basename(filepath)} ---")
        print(f"  Total samples loaded: {len(data)}")
        print(f"  Records processed: {record_count}")
        print(f"  Block length: {header.get('blockLength', 'N/A')}")
        print(f"  Expected samples: {record_count * header.get('blockLength', 0)}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(data) / sample_rate:.3f} seconds")
        print(f"  Timestamp range: {full_timestamps[0]:.3f} to {full_timestamps[-1]:.3f}s")
        print("-" * 60 + "\n")
        
    return result

class MultiChannelLoadThread(QThread):
    """Thread for loading multiple channels with absolute timestamps"""
    
    progress_update = pyqtSignal(int, str)
    loading_complete = pyqtSignal(object, object, object, str)
    loading_error = pyqtSignal(str)
    
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        
    def run(self):
        """Load files with proper timestamp handling"""
        try:
            all_data = []
            all_timestamps = []
            headers = []
            file_names = []
            
            total_files = len(self.file_paths)
            
            print(f"Loading {total_files} files...")
            
            # Load each file
            for i, file_path in enumerate(self.file_paths):
                file_name = os.path.basename(file_path)
                
                # Update progress
                progress = int((i / total_files) * 90)
                self.progress_update.emit(progress, f"Loading {file_name}...")
                
                # Load the file with timestamps
                try:
                    result = load_continuous_with_timestamps(file_path, verbose=True)
                    
                    all_data.append(result['data'])
                    all_timestamps.append(result['timestamps'])
                    headers.append(result['header'])
                    file_names.append(file_name)
                    
                except Exception as e:
                    raise Exception(f"Failed to load {file_name}: {str(e)}")
            
            # Validation phase
            self.progress_update.emit(90, "Validating data compatibility...")
            
            # Check data lengths
            lengths = [len(data) for data in all_data]
            min_length = min(lengths)
            max_length = max(lengths)
            
            if max_length - min_length > 1000:  # Allow small differences
                print(f"Warning: File lengths vary significantly: {dict(zip(file_names, lengths))}")
                # Trim all to minimum length
                for i in range(len(all_data)):
                    all_data[i] = all_data[i][:min_length]
                    all_timestamps[i] = all_timestamps[i][:min_length]
                print(f"Trimmed all files to {min_length} samples")
            
            # Check sample rates
            sample_rates = [header.get('sampleRate', 30000) for header in headers]
            unique_rates = set(sample_rates)
            if len(unique_rates) > 1:
                print(f"Warning: Different sample rates detected: {dict(zip(file_names, sample_rates))}")
                # Use the most common sample rate
                from collections import Counter
                most_common_rate = Counter(sample_rates).most_common(1)[0][0]
                print(f"Using sample rate: {most_common_rate} Hz")
            else:
                most_common_rate = sample_rates[0]
            
            # Use timestamps from the first file as reference
            # This assumes all files are from the same recording session
            reference_timestamps = all_timestamps[0]
            
            # Create multi-channel array
            self.progress_update.emit(95, "Creating multi-channel array...")
            
            if len(all_data) == 1:
                multi_channel_data = all_data[0].reshape(-1, 1)
            else:
                # Ensure all data has same length
                final_length = min(len(data) for data in all_data)
                trimmed_data = [data[:final_length] for data in all_data]
                multi_channel_data = np.column_stack(trimmed_data)
                reference_timestamps = reference_timestamps[:final_length]
            
            # Create combined header
            combined_header = headers[0].copy()
            combined_header.update({
                'channel_files': file_names,
                'channel_count': len(file_names),
                'sampleRate': most_common_rate,
                'multi_channel': True,
                'original_headers': headers
            })
            print("\n" + "="*60)
            print("DEBUG: MultiChannelLoadThread.run() - HEADER CHECK")
            print("="*60)
            print(f"file_names: {file_names}")
            print(f"combined_header keys: {list(combined_header.keys())}")
            print(f"'channel_files' in header: {'channel_files' in combined_header}")
            print(f"channel_files value: {combined_header.get('channel_files')}")
            print("="*60 + "\n")
            
            description = f"Multi-channel ({len(file_names)} channels)"
            
            self.progress_update.emit(100, "Loading complete!")
            
            print(f"Multi-channel loading complete:")
            print(f"  Final shape: {multi_channel_data.shape}")
            print(f"  Time range: {reference_timestamps[0]:.3f} to {reference_timestamps[-1]:.3f} seconds")
            print(f"  Sample rate: {most_common_rate} Hz")
            
            self.loading_complete.emit(multi_channel_data, reference_timestamps, combined_header, description)
            
        except Exception as e:
            error_msg = f"Multi-channel loading failed: {str(e)}"
            print(error_msg)
            self.loading_error.emit(error_msg)

class FileSelectionDialog(QDialog):
    """Simple dialog showing selected files before loading"""
    
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.setWindowTitle("Confirm Multi-Channel Loading")
        self.setModal(True)
        self.resize(500, 400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(f"Loading {len(self.file_paths)} .continuous files:")
        layout.addWidget(info_label)
        
        # File list
        file_list = QListWidget()
        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)
            item = QListWidgetItem(file_name)
            file_list.addItem(item)
        layout.addWidget(file_list)
        
        # Warning label
        warning_label = QLabel(
            "Note: Files will be synchronized using timestamps.\n"
            "Make sure all files are from the same recording session."
        )
        warning_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(warning_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("Load Files")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)

class MultiChannelLoader:
    """Main class for handling multi-channel file loading with absolute time"""
    
    def __init__(self, parent_window):
        self.parent = parent_window
        self.load_thread = None
        self.progress_dialog = None
    
    def select_and_load_files(self, callback_function):
        """Show file dialog and load multiple channels"""
        # File selection dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent,
            'Select multiple .continuous files',
            filter="Continuous files (*.continuous);;All files (*.*)"
        )
        
        if not file_paths:
            return
        
        # Validate minimum files
        if len(file_paths) < 1:
            QMessageBox.warning(
                self.parent,
                "No Files Selected",
                "Please select at least one .continuous file."
            )
            return
        
        # Show file confirmation dialog
        confirm_dialog = FileSelectionDialog(file_paths, self.parent)
        if confirm_dialog.exec_() != QDialog.Accepted:
            return
        
        # Validate file extensions
        invalid_files = [f for f in file_paths if not f.lower().endswith('.continuous')]
        if invalid_files:
            QMessageBox.warning(
                self.parent,
                "Invalid Files",
                f"Some files are not .continuous files:\n" +
                "\n".join([os.path.basename(f) for f in invalid_files[:5]]) +
                ("..." if len(invalid_files) > 5 else "")
            )
            return
        
        # Start loading
        self.load_files_threaded(file_paths, callback_function)
    
    def load_files_threaded(self, file_paths, callback_function):
        """Load files in background thread"""
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog(
            "Loading multi-channel data...", 
            "Cancel", 
            0, 100, 
            self.parent
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setMinimumDuration(500)  # Show after 500ms
        
        # Create loading thread
        self.load_thread = MultiChannelLoadThread(file_paths)
        
        # Connect signals
        self.load_thread.progress_update.connect(self.update_progress)
        self.load_thread.loading_complete.connect(
            lambda data, timestamps, header, desc: self.on_loading_complete(
                data, timestamps, header, desc, callback_function
            )
        )
        self.load_thread.loading_error.connect(self.on_loading_error)
        self.progress_dialog.canceled.connect(self.cancel_loading)
        
        # Start loading
        self.load_thread.start()
        self.progress_dialog.show()
    
    def update_progress(self, percent, message):
        """Update progress dialog"""
        if self.progress_dialog:
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
    
    def on_loading_complete(self, data, timestamps, header, description, callback_function):
        """Handle successful loading completion"""
        if self.progress_dialog:
            self.progress_dialog.close()
        
        # Show summary
        file_names = header.get('channel_files', [])
        sample_rate = header.get('sampleRate', 'Unknown')
        duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 0 else 0
        
        QMessageBox.information(
            self.parent,
            "Multi-Channel Load Complete",
            f"Successfully loaded {len(file_names)} channels:\n\n" +
            "\n".join([f"• {name}" for name in file_names[:8]]) +
            ("..." if len(file_names) > 8 else "") +
            f"\n\nData shape: {data.shape}\n" +
            f"Sample rate: {sample_rate} Hz\n" +
            f"Duration: {duration:.2f} seconds\n"
        )
        
        # Call the callback function
        callback_function(data, timestamps, header, description)
    
    def on_loading_error(self, error_message):
        """Handle loading errors"""
        if self.progress_dialog:
            self.progress_dialog.close()
        
        QMessageBox.critical(
            self.parent,
            "Multi-Channel Loading Error",
            f"Failed to load multi-channel data:\n\n{error_message}\n\n"
            f"Tips:\n"
            f"• Ensure all files are valid .continuous files\n"
            f"• Check that files are from the same recording session\n"
            f"• Verify files are not corrupted"
        )
    
    def cancel_loading(self):
        """Cancel the loading process"""
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.terminate()
            self.load_thread.wait(3000)  # Wait up to 3 seconds
        
        if self.progress_dialog:
            self.progress_dialog.close()
        
        print("Multi-channel loading cancelled by user")

# Convenience functions
def load_single_channel(file_path, verbose=True):
    """Load a single channel file with absolute timestamps"""
    return load_continuous_with_timestamps(file_path, verbose=verbose)

def load_multiple_channels_dialog(parent_window, callback_function):
    """Convenience function to show multi-channel loading dialog"""
    loader = MultiChannelLoader(parent_window)
    loader.select_and_load_files(callback_function)


   