"""
Dialog Components for Neural Data Analysis
Complete dialog collection with all required classes
"""
import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                             QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
                             QListWidget, QListWidgetItem, QTextEdit,
                             QFileDialog, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class PreprocessingDialog(QDialog):
    """Enhanced preprocessing dialog with parameter validation"""
    
    def __init__(self, data, header, timestamps, parent=None):
        super().__init__(parent)
        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.original_fs = header.get('sampleRate')
        if self.original_fs is None or self.original_fs <= 0:
            QMessageBox.critical(self, "Invalid File", 
                                "File missing valid sampling rate information")
            return
        
        self.setWindowTitle("Preprocess Neural Data")
        self.setModal(True)
        self.resize(600, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Data info
        info_text = f"Data: {self.data.shape} • Rate: {self.original_fs:.0f} Hz"
        if self.timestamps is not None:
            duration = self.timestamps[-1] - self.timestamps[0]
            info_text += f" • Duration: {duration:.1f}s"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("padding: 8px; background: #f0f8ff; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(info_label)
        
        # Channel selection
        self.setup_channel_selection(layout)
        
        # Sampling rate
        self.setup_sampling_rate(layout)
        
        # Filters
        self.setup_filters(layout)
        
        # Summary
        self.setup_summary(layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect signals
        self.connect_signals()
        self.update_summary()
    
    def setup_channel_selection(self, layout):
        """Setup channel selection"""
        group = QGroupBox("Channel Selection (max 8)")
        group_layout = QVBoxLayout(group)
        
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.MultiSelection)
        self.channel_list.setMaximumHeight(100)
        
        n_channels = self.data.shape[1] if self.data.ndim > 1 else 1
        
        # Get channel names
        if 'channel_files' in self.header and self.header['channel_files']:
            channel_names = self.header['channel_files']
        else:
            channel_names = [f"Channel {i+1}" for i in range(n_channels)]
        
        # Add channels to list
        for i in range(min(n_channels, 16)):
            name = channel_names[i] if i < len(channel_names) else f"Channel {i+1}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)
            self.channel_list.addItem(item)
            if i < 4:  # Select first 4 by default
                item.setSelected(True)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all_channels)
        btn_layout.addWidget(select_all)
        
        clear_all = QPushButton("Clear")
        clear_all.clicked.connect(self.clear_channels)
        btn_layout.addWidget(clear_all)
        
        group_layout.addWidget(self.channel_list)
        group_layout.addLayout(btn_layout)
        layout.addWidget(group)
    
    def setup_sampling_rate(self, layout):
        """Setup sampling rate controls"""
        group = QGroupBox("Sampling Rate")
        group_layout = QGridLayout(group)
        
        # Show current rate 
        current_rate = self.header.get('sampleRate')
        if current_rate is None:
            QMessageBox.critical(self, "Error", "Cannot determine sampling rate from data")
            return

        
        info_label = QLabel(f"Current rate: {current_rate:.0f} Hz (already downsampled)")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        group_layout.addWidget(info_label, 0, 0)
        
        # Hide resampling controls
        self.enable_resample = QCheckBox("Enable resampling")
        self.enable_resample.setChecked(False)  # Disabled
        self.enable_resample.setVisible(False)  # Hidden
        
        self.target_fs = QComboBox()
        self.target_fs.setCurrentText(str(int(current_rate)))
        self.target_fs.setVisible(False)  # Hidden
        
        layout.addWidget(group)
    
    def setup_filters(self, layout):
        """Setup filter controls"""
        group = QGroupBox("Filters")
        group_layout = QGridLayout(group)
        
        # Notch filter
        self.enable_notch = QCheckBox("50Hz Notch filter")
        self.enable_notch.setChecked(True)
        group_layout.addWidget(self.enable_notch, 0, 0, 1, 2)
        
        # Bandpass filter
        self.enable_bandpass = QCheckBox("Bandpass filter")
        self.enable_bandpass.setChecked(True)
        group_layout.addWidget(self.enable_bandpass, 1, 0, 1, 2)
        
        # Frequency settings with FIXED ranges to avoid overflow
        group_layout.addWidget(QLabel("Low cutoff (Hz):"), 2, 0)
        self.low_cutoff = QDoubleSpinBox()
        self.low_cutoff.setRange(0.1, 500.0)  # Fixed: reasonable range
        self.low_cutoff.setValue(80.0)
        self.low_cutoff.setSingleStep(10.0)
        group_layout.addWidget(self.low_cutoff, 2, 1)
        
        group_layout.addWidget(QLabel("High cutoff (Hz):"), 3, 0)
        self.high_cutoff = QDoubleSpinBox()
        self.high_cutoff.setRange(1.0, 1000.0)  # Fixed: reasonable range
        self.high_cutoff.setValue(250.0)
        self.high_cutoff.setSingleStep(10.0)
        group_layout.addWidget(self.high_cutoff, 3, 1)
        
        # Presets
        preset_layout = QHBoxLayout()
        
        ripple_btn = QPushButton("Ripples (80-250Hz)")
        ripple_btn.clicked.connect(lambda: self.set_preset(80, 250))
        preset_layout.addWidget(ripple_btn)
        
        gamma_btn = QPushButton("Gamma (30-100Hz)")
        gamma_btn.clicked.connect(lambda: self.set_preset(30, 100))
        preset_layout.addWidget(gamma_btn)
        
        group_layout.addLayout(preset_layout, 4, 0, 1, 2)
        layout.addWidget(group)
    
    def setup_summary(self, layout):
        """Setup summary display"""
        group = QGroupBox("Processing Summary")
        group_layout = QVBoxLayout(group)
        
        self.summary_label = QLabel("Configure parameters above")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("padding: 8px; background: #f8f9fa; border: 1px solid #ccc; border-radius: 4px;")
        group_layout.addWidget(self.summary_label)
        
        layout.addWidget(group)
    
    def connect_signals(self):
        """Connect UI signals"""
        self.channel_list.itemSelectionChanged.connect(self.update_summary)
        self.enable_resample.toggled.connect(self.update_summary)
        self.target_fs.currentTextChanged.connect(self.update_summary)
        self.enable_bandpass.toggled.connect(self.update_summary)
        self.enable_notch.toggled.connect(self.update_summary)
        self.low_cutoff.valueChanged.connect(self.update_summary)
        self.high_cutoff.valueChanged.connect(self.update_summary)
    
    def select_all_channels(self):
        """Select all channels"""
        for i in range(self.channel_list.count()):
            self.channel_list.item(i).setSelected(True)
    
    def clear_channels(self):
        """Clear all channel selections"""
        self.channel_list.clearSelection()
    
    def set_preset(self, low, high):
        """Set filter preset"""
        self.low_cutoff.setValue(low)
        self.high_cutoff.setValue(high)
        self.enable_bandpass.setChecked(True)
    
    def update_summary(self):
        """Update processing summary"""
        channels = self.get_selected_channels()
        
        if not channels:
            self.summary_label.setText("Please select at least one channel")
            return
        
        summary = []
        summary.append(f"Channels: {len(channels)} selected")
        
        if self.enable_resample.isChecked():
            try:
                target_fs = float(self.target_fs.currentText())
                ratio = self.original_fs / target_fs
                summary.append(f"Resample: {self.original_fs:.0f}Hz → {target_fs:.0f}Hz ({ratio:.1f}x)")
            except:
                summary.append("Resample: Invalid target frequency")
        else:
            summary.append(f"No resampling ({self.original_fs:.0f}Hz)")
        
        filters = []
        if self.enable_notch.isChecked():
            filters.append("50Hz notch")
        if self.enable_bandpass.isChecked():
            filters.append(f"{self.low_cutoff.value():.0f}-{self.high_cutoff.value():.0f}Hz bandpass")
        
        if filters:
            summary.append(f"Filters: {', '.join(filters)}")
        else:
            summary.append("No filters")
        
        if self.timestamps is not None:
            duration = self.timestamps[-1] - self.timestamps[0]
            summary.append(f"Duration: {duration:.1f}s (absolute timing preserved)")
        
        self.summary_label.setText("\n".join(summary))
    
    def get_selected_channels(self):
        """Get selected channel indices"""
        selected = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.isSelected():
                selected.append(item.data(Qt.UserRole))
        return selected
    
    def get_parameters(self):
        """Get processing parameters"""
        target_fs = self.original_fs
        if self.enable_resample.isChecked():
            try:
                target_fs = float(self.target_fs.currentText())
            except:
                target_fs = 1000
        
        return {
            'channels': self.get_selected_channels(),
            'target_fs': target_fs,
            'notch_enabled': self.enable_notch.isChecked(),
            'bandpass_enabled': self.enable_bandpass.isChecked(),
            'low_cutoff': self.low_cutoff.value(),
            'high_cutoff': self.high_cutoff.value(),
            'original_fs': self.original_fs
        }
    
    def accept(self):
        """Validate and accept"""
        if not self.get_selected_channels():
            QMessageBox.warning(self, "No Channels", "Please select at least one channel")
            return
        
        params = self.get_parameters()
        if params['bandpass_enabled'] and params['low_cutoff'] >= params['high_cutoff']:
            QMessageBox.warning(self, "Invalid Filter", "Low cutoff must be less than high cutoff")
            return
        
        super().accept()

class ChannelSelectionDialog(QDialog):
    """Simple channel selection dialog"""
    
    def __init__(self, channel_names, selected_channels=None, parent=None):
        super().__init__(parent)
        self.channel_names = channel_names
        self.setWindowTitle("Channel Selection")
        self.setModal(True)
        self.resize(400, 400)
        self.setup_ui()
        self.populate_channels(selected_channels)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Select channels to display:"))
        
        self.channel_tree = QTreeWidget()
        self.channel_tree.setHeaderLabels(["Channel", "Index"])
        self.channel_tree.setRootIsDecorated(False)
        layout.addWidget(self.channel_tree)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all)
        btn_layout.addWidget(select_all)
        
        clear_all = QPushButton("Clear")
        clear_all.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_all)
        
        layout.addLayout(btn_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def populate_channels(self, selected_channels=None):
        if selected_channels is None:
            selected_channels = list(range(min(len(self.channel_names), 8)))
        
        self.channel_items = []
        for i, name in enumerate(self.channel_names):
            item = QTreeWidgetItem([name, str(i)])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if i in selected_channels else Qt.Unchecked)
            self.channel_tree.addTopLevelItem(item)
            self.channel_items.append(item)
    
    def select_all(self):
        for item in self.channel_items:
            item.setCheckState(0, Qt.Checked)
    
    def clear_all(self):
        for item in self.channel_items:
            item.setCheckState(0, Qt.Unchecked)
    
    def get_selected_channels(self):
        return [i for i, item in enumerate(self.channel_items) 
                if item.checkState(0) == Qt.Checked]

class RippleDetectionDialog(QDialog):
    """Simple ripple detection dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ripple Detection")
        self.setModal(True)
        self.resize(350, 250)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Frequency band
        freq_group = QGroupBox("Frequency Band")
        freq_layout = QGridLayout(freq_group)
        
        self.low_freq = QSpinBox()
        self.low_freq.setRange(1, 500)  # Fixed range
        self.low_freq.setValue(80)
        freq_layout.addWidget(QLabel("Low Frequency (Hz):"), 0, 0)
        freq_layout.addWidget(self.low_freq, 0, 1)
        
        self.high_freq = QSpinBox()
        self.high_freq.setRange(50, 1000)  # Fixed range
        self.high_freq.setValue(250)
        freq_layout.addWidget(QLabel("High Frequency (Hz):"), 1, 0)
        freq_layout.addWidget(self.high_freq, 1, 1)
        
        layout.addWidget(freq_group)
        
        # Detection parameters
        detect_group = QGroupBox("Detection")
        detect_layout = QGridLayout(detect_group)
        
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 10.0)
        self.threshold.setValue(3.0)
        self.threshold.setSingleStep(0.1)
        detect_layout.addWidget(QLabel("Threshold (SD):"), 0, 0)
        detect_layout.addWidget(self.threshold, 0, 1)
        
        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(1.0, 100.0)
        self.min_duration.setValue(15.0)
        detect_layout.addWidget(QLabel("Min Duration (ms):"), 1, 0)
        detect_layout.addWidget(self.min_duration, 1, 1)
        
        layout.addWidget(detect_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_parameters(self):
        return {
            'low_freq': self.low_freq.value(),
            'high_freq': self.high_freq.value(),
            'threshold': self.threshold.value(),
            'min_duration': self.min_duration.value() / 1000.0
        }

class FilterDialog(QDialog):
    """Simple filter dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Settings")
        self.setModal(True)
        self.resize(300, 200)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Filter type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Bandpass", "Lowpass", "Highpass", "Notch"])
        type_layout.addWidget(self.filter_type)
        layout.addLayout(type_layout)
        
        # Frequencies
        freq_layout = QGridLayout()
        
        self.low_freq = QDoubleSpinBox()
        self.low_freq.setRange(0.1, 500.0)  # Fixed range
        self.low_freq.setValue(1.0)
        freq_layout.addWidget(QLabel("Low Frequency (Hz):"), 0, 0)
        freq_layout.addWidget(self.low_freq, 0, 1)
        
        self.high_freq = QDoubleSpinBox()
        self.high_freq.setRange(0.1, 1000.0)  # Fixed range
        self.high_freq.setValue(100.0)
        freq_layout.addWidget(QLabel("High Frequency (Hz):"), 1, 0)
        freq_layout.addWidget(self.high_freq, 1, 1)
        
        layout.addLayout(freq_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_filter_params(self):
        return {
            'type': self.filter_type.currentText().lower(),
            'low_freq': self.low_freq.value(),
            'high_freq': self.high_freq.value()
        }

class ExportDialog(QDialog):
    """Simple export dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.setModal(True)
        self.resize(350, 200)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Options
        self.export_raw = QCheckBox("Raw data")
        self.export_raw.setChecked(True)
        layout.addWidget(self.export_raw)
        
        self.export_filtered = QCheckBox("Filtered data")
        layout.addWidget(self.export_filtered)
        
        # Format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "NumPy (.npy)", "MATLAB (.mat)"])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        # File path
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "All Files (*)")
        if file_path:
            self.file_path.setText(file_path)
    
    def get_export_settings(self):
        return {
            'export_raw': self.export_raw.isChecked(),
            'export_filtered': self.export_filtered.isChecked(),
            'format': self.format_combo.currentText(),
            'file_path': self.file_path.text()
        }

class ProgressDialog(QDialog):
    """Simple progress dialog for long operations"""
    
    def __init__(self, title="Processing", description="Please wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 120)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setup_ui(description)
    
    def setup_ui(self, description):
        layout = QVBoxLayout(self)
        
        # Description
        self.description_label = QLabel(description)
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
    
    def update_progress(self, value, status=""):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
    
    def set_description(self, description):
        """Update description text"""
        self.description_label.setText(description)

class DownsampleDialog(QDialog):
    """Dynamic downsampling dialog with smart frequency suggestions"""
    
    def __init__(self, data, header, timestamps, file_path, parent=None):
        super().__init__(parent)
        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.file_path = file_path  # Store file_path
        self.original_fs = header.get('sampleRate')
        if self.original_fs is None or self.original_fs <= 0:
            QMessageBox.critical(self, "Invalid File", 
                                "File missing valid sampling rate information")
            return
        
        self.setWindowTitle("Downsample Data")
        self.setModal(True)
        self.resize(400, 350)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File info
        filename = os.path.basename(str(self.file_path)) if self.file_path else "Unknown file"
        info_text = f"File: {filename}\n"
        info_text += f"Original sampling rate: {self.original_fs:.0f} Hz\n"
        info_text += f"Samples: {len(self.data):,}\n"
        
        if self.timestamps is not None and len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0] 
            info_text += f"Duration: {duration:.1f} seconds\n"
            start_time = self.timestamps[0]
            info_text += f"Start time: {start_time:.3f} seconds"
        else:
            duration = len(self.data) / self.original_fs
            info_text += f"Duration: {duration:.1f} seconds"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        layout.addWidget(info_label)
        
        # Downsample section
        downsample_group = QGroupBox("Downsample Settings")
        downsample_layout = QVBoxLayout(downsample_group)
        
        # Explanation
        explain_label = QLabel(
            "Choose target sampling rate for analysis.\n"
            "Lower rates improve performance and are suitable for most analyses."
        )
        explain_label.setWordWrap(True)
        explain_label.setStyleSheet("color: #666; margin: 5px 0;")
        downsample_layout.addWidget(explain_label)
        
        # Frequency selection
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Target frequency (Hz):"))
        
        self.freq_combo = QComboBox()
        self.freq_combo.setEditable(True)
        
        # Smart suggestions based on original frequency
        suggestions = self.get_smart_suggestions()
        self.freq_combo.addItems([str(f) for f in suggestions])
        
        # Set default to 1000Hz or closest available
        default_freq = 1000 if 1000 in suggestions else suggestions[0]
        self.freq_combo.setCurrentText(str(default_freq))
        
        freq_layout.addWidget(self.freq_combo)
        downsample_layout.addLayout(freq_layout)
        
        # Recommendations
        rec_layout = QVBoxLayout()
        
        rec_label = QLabel("Recommendations:")
        rec_label.setFont(QFont("Arial", 9, QFont.Bold))
        rec_layout.addWidget(rec_label)
        
        recommendations = self.get_recommendations()
        for rec in recommendations:
            rec_item = QLabel(f"• {rec}")
            rec_item.setStyleSheet("color: #495057; margin-left: 10px; font-size: 9pt;")
            rec_item.setWordWrap(True)
            rec_layout.addWidget(rec_item)
        
        downsample_layout.addLayout(rec_layout)
        layout.addWidget(downsample_group)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 3px;
                font-family: monospace;
                font-size: 9pt;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect signals
        self.freq_combo.currentTextChanged.connect(self.update_preview)
        self.update_preview()
    
    def get_smart_suggestions(self):
        """Generate smart frequency suggestions based on original rate"""
        suggestions = []
        
        # Always include these common rates if they make sense
        common_rates = [500, 1000, 2000, 5000, 10000, 20000]
        
        for rate in common_rates:
            if rate < self.original_fs * 0.9:  # Must be significantly lower
                suggestions.append(rate)
        
        # Add some fractions of original rate
        fractions = [0.1, 0.2, 0.25, 0.5]
        for frac in fractions:
            suggested = int(self.original_fs * frac)
            if suggested >= 500 and suggested not in suggestions:
                suggestions.append(suggested)
        
        # Sort and ensure we have at least a few options
        suggestions = sorted(set(suggestions))
        
        # Ensure we have at least 3 suggestions
        if len(suggestions) < 3:
            if 1000 not in suggestions and 1000 < self.original_fs:
                suggestions.append(1000)
            if 2000 not in suggestions and 2000 < self.original_fs:
                suggestions.append(2000)
            if 500 not in suggestions:
                suggestions.append(500)
        
        return sorted(set(suggestions))
    
    def get_recommendations(self):
        """Get recommendations based on original sampling rate"""
        recommendations = []
        
        if self.original_fs >= 30000:
            recommendations.extend([
                "1000-2000Hz: Excellent for ripples (80-250Hz), theta, gamma",
                "5000Hz: Good for high-frequency oscillations up to ~2000Hz",
                "10000Hz: Preserves most neural signals, larger file size"
            ])
        elif self.original_fs >= 20000:
            recommendations.extend([
                "1000Hz: Recommended for ripple analysis (80-250Hz)",
                "2000-5000Hz: Good for broader frequency analysis"
            ])
        elif self.original_fs >= 10000:
            recommendations.extend([
                "1000-2000Hz: Good for most LFP analysis",
                f"{int(self.original_fs * 0.5)}Hz: Preserves most content"
            ])
        else:
            recommendations.append(f"Consider {int(self.original_fs * 0.5)}Hz to reduce file size")
        
        # Add general recommendations
        if self.original_fs > 5000:
            recommendations.append("Lower rates improve processing speed and memory usage")
            
        return recommendations
    
    def update_preview(self):
        """Update the preview based on selected frequency"""
        try:
            target_fs = float(self.freq_combo.currentText())
            
            if target_fs >= self.original_fs:
                self.preview_label.setText("Target must be less than original rate")
                self.preview_label.setStyleSheet("""
                    QLabel {
                        padding: 8px; background: #f8d7da; border: 1px solid #f5c6cb;
                        border-radius: 3px; color: #721c24; font-family: monospace; font-size: 9pt;
                    }
                """)
                return
            
            # Calculate new parameters
            downsample_factor = self.original_fs / target_fs
            new_samples = int(len(self.data) / downsample_factor)
            
            if self.timestamps is not None:
                duration = self.timestamps[-1] - self.timestamps[0]
            else:
                duration = len(self.data) / self.original_fs
            
            # Calculate file size reduction
            size_reduction = downsample_factor
            
            # Nyquist frequency info
            nyquist_orig = self.original_fs / 2
            nyquist_new = target_fs / 2
            
            preview_text = f"Downsample: {self.original_fs:.0f}Hz → {target_fs:.0f}Hz\n"
            preview_text += f"Factor: {downsample_factor:.1f}x reduction\n"
            preview_text += f"Samples: {len(self.data):,} → {new_samples:,}\n"
            preview_text += f"Duration: {duration:.1f}s (preserved)\n"
            preview_text += f"File size: ~{size_reduction:.1f}x smaller\n"
            preview_text += f"Nyquist: {nyquist_orig:.0f}Hz → {nyquist_new:.0f}Hz\n"
            
            # Add frequency analysis info
            if target_fs >= 2000:
                preview_text += "Suitable for ripples (80-250Hz)"
            elif target_fs >= 1000:
                preview_text += "Good for ripples, may limit high-gamma"
            else:
                preview_text += "May limit some high-frequency content"
            
            self.preview_label.setText(preview_text)
            self.preview_label.setStyleSheet("""
                QLabel {
                    padding: 8px; background: #d1ecf1; border: 1px solid #bee5eb;
                    border-radius: 3px; color: #0c5460; font-family: monospace; font-size: 9pt;
                }
            """)
            
        except ValueError:
            self.preview_label.setText("Please enter a valid frequency")
            self.preview_label.setStyleSheet("""
                QLabel {
                    padding: 8px; background: #fff3cd; border: 1px solid #ffeaa7;
                    border-radius: 3px; color: #856404; font-family: monospace; font-size: 9pt;
                }
            """)
    
    def get_target_frequency(self):
        """Get and validate target frequency"""
        try:
            target = float(self.freq_combo.currentText())
            if target >= self.original_fs:
                QMessageBox.warning(self, "Invalid Frequency", 
                                f"Target frequency ({target:.0f}Hz) must be less than original ({self.original_fs:.0f}Hz)")
                return None
            if target < 100:
                reply = QMessageBox.question(self, "Very Low Frequency", 
                    f"Target frequency ({target:.0f}Hz) is very low.\n"
                    f"This may limit analysis capabilities.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return None
            return target
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid numeric frequency")
            return None

class MultiChannelDownsampleDialog(QDialog):
    """Combined dialog for multichannel confirmation and downsampling"""
    
    def __init__(self, data, timestamps, header, description, parent=None):
        super().__init__(parent)
        self.data = data
        self.timestamps = timestamps
        self.header = header
        self.description = description
        self.original_fs = header.get('sampleRate')
        if self.original_fs is None or self.original_fs <= 0:
            QMessageBox.critical(self, "Invalid File", 
                                "File missing valid sampling rate information")
            return
        
        self.setWindowTitle("Multi-Channel Data Loading")
        self.setModal(True)
        self.resize(600, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Add the missing title_label definition
        title_label = QLabel("Multi-Channel Data Loading & Downsampling")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 10px; background: #f8f9fa; border: 1px solid #ddd;")
    
        layout.addWidget(title_label)
    
    # Section 1: Multi-channel confirmation
        self.setup_channel_confirmation(layout)
        # Section 2: Downsample settings
        self.setup_downsample_section(layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Load & Downsample")
        layout.addWidget(buttons)
        
        # Connect signals
        self.freq_combo.currentTextChanged.connect(self.update_preview)
        self.update_preview()
    
    def setup_channel_confirmation(self, layout):
        """Setup the multichannel confirmation section"""
        confirm_group = QGroupBox("Loaded Files Confirmation")
        confirm_group.setFont(QFont("Arial", 12, QFont.Bold))
        confirm_layout = QVBoxLayout(confirm_group)
        
        # Data summary
        n_channels = self.data.shape[1] if self.data.ndim > 1 else 1
        n_samples = len(self.data)
        duration = (self.timestamps[-1] - self.timestamps[0]) if self.timestamps is not None else (n_samples / self.original_fs)
        
        summary_text = f"Successfully loaded {n_channels} channels\n"
        summary_text += f"Total samples: {n_samples:,}\n"
        summary_text += f"Original sampling rate: {self.original_fs:.0f} Hz\n"
        summary_text += f"Duration: {duration:.2f} seconds\n"
        summary_text += f"Memory usage: {self.data.nbytes / 1024 / 1024:.1f} MB"
        
        summary_label = QLabel(summary_text)
        summary_label.setStyleSheet("""
            QLabel {
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
                color: #0c5460;
            }
        """)
        confirm_layout.addWidget(summary_label)
        
        # Channel files list
        if 'channel_files' in self.header and self.header['channel_files']:
            files_label = QLabel("Channel Files:")
            files_label.setFont(QFont("Arial", 10, QFont.Bold))
            confirm_layout.addWidget(files_label)
            
            self.files_list = QListWidget()
            self.files_list.setMaximumHeight(120)
            
            for i, filename in enumerate(self.header['channel_files']):
                clean_name = os.path.basename(filename)
                item = QListWidgetItem(f"CH{i+1}: {clean_name}")
                item.setToolTip(filename)
                self.files_list.addItem(item)
            
            confirm_layout.addWidget(self.files_list)
        
        layout.addWidget(confirm_group)
    
    def setup_downsample_section(self, layout):
        """Setup the downsample section"""
        downsample_group = QGroupBox("Downsample Settings")
        downsample_group.setFont(QFont("Arial", 12, QFont.Bold))
        downsample_layout = QVBoxLayout(downsample_group)
        
        # Explanation
        explain_text = ("Choose target sampling rate for efficient analysis.\n"
                       "Lower rates improve performance and are suitable for most neural analyses.")
        explain_label = QLabel(explain_text)
        explain_label.setWordWrap(True)
        explain_label.setStyleSheet("color: #666; margin: 5px 0; padding: 8px;")
        downsample_layout.addWidget(explain_label)
        
        # Frequency selection
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Target frequency (Hz):"))
        
        self.freq_combo = QComboBox()
        self.freq_combo.setEditable(True)
        
        # Smart suggestions
        suggestions = self.get_smart_suggestions()
        self.freq_combo.addItems([str(f) for f in suggestions])
        
        # Set default
        default_freq = 1000 if 1000 in suggestions else suggestions[0]
        self.freq_combo.setCurrentText(str(default_freq))
        
        freq_layout.addWidget(self.freq_combo)
        downsample_layout.addLayout(freq_layout)
        
        # Preview section
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("""
            QLabel {
                background: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
                font-size: 9pt;
            }
        """)
        downsample_layout.addWidget(self.preview_label)
        
        layout.addWidget(downsample_group)
    
    def get_smart_suggestions(self):
        """Generate smart frequency suggestions"""
        suggestions = []
        common_rates = [500, 1000, 2000, 5000, 10000]
        
        for rate in common_rates:
            if rate < self.original_fs * 0.9:
                suggestions.append(rate)
        
        # Add fractions of original rate
        for frac in [0.1, 0.2, 0.25, 0.5]:
            suggested = int(self.original_fs * frac)
            if suggested >= 500 and suggested not in suggestions:
                suggestions.append(suggested)
        
        suggestions = sorted(set(suggestions))
        
        # Ensure minimum options
        if len(suggestions) < 3:
            for rate in [500, 1000, 2000]:
                if rate not in suggestions and rate < self.original_fs:
                    suggestions.append(rate)
        
        return sorted(set(suggestions))
    
    
    def update_preview(self):
        """Update preview based on selected frequency"""
        try:
            target_fs = float(self.freq_combo.currentText())
            
            if target_fs >= self.original_fs:
                self.preview_label.setText("Target must be less than original rate")
                self.preview_label.setStyleSheet("""
                    QLabel {
                        background: #f8d7da; border: 1px solid #f5c6cb;
                        border-radius: 5px; padding: 10px; color: #721c24;
                        font-family: monospace; font-size: 9pt;
                    }
                """)
                return
            
            
        except ValueError:
            self.preview_label.setText("Please enter a valid frequency")
            self.preview_label.setStyleSheet("""
                QLabel {
                    background: #fff3cd; border: 1px solid #ffeaa7;
                    border-radius: 5px; padding: 10px; color: #856404;
                    font-family: monospace; font-size: 9pt;
                }
            """)
    
    def get_target_frequency(self):
        """Get and validate target frequency"""
        try:
            target = float(self.freq_combo.currentText())
            if target >= self.original_fs:
                return None
            if target < 100:
                return None
            return target
        except ValueError:
            return None

# Test the dialogs
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    print("Testing dialog components...")
    
    # Test ProgressDialog
    progress = ProgressDialog("Test Progress", "Testing progress dialog")
    progress.update_progress(50, "Processing...")
    print("✓ ProgressDialog created successfully")
    
    print("All dialog components loaded successfully!")
    print("Available dialogs:")
    print("• PreprocessingDialog")
    print("• ChannelSelectionDialog") 
    print("• RippleDetectionDialog")
    print("• FilterDialog")
    print("• ExportDialog")
    print("• ProgressDialog")
    print("• DownsampleDialog")
    print("• MultiChannelDownsampleDialog")