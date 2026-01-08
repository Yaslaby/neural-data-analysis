"""
Dialog Components for Neural Data Analysis
"""
import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QComboBox, QDoubleSpinBox,
                             QCheckBox, QGroupBox, QDialogButtonBox,
                             QTreeWidget, QTreeWidgetItem, QListWidget,
                             QListWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class PreprocessingDialog(QDialog):
    """Preprocessing dialog with channel selection and filter options"""
    
    def __init__(self, data, header, timestamps, parent=None):
        super().__init__(parent)
        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.original_fs = header.get('sampleRate')
        self.is_preprocessed = header.get('preprocessed', False)
        
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
        info_text = f"Data: {self.data.shape} - Rate: {self.original_fs:.0f} Hz"
        if self.timestamps is not None:
            duration = self.timestamps[-1] - self.timestamps[0]
            info_text += f" - Duration: {duration:.1f}s"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("padding: 8px; background: #f0f8ff; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(info_label)
        
        # Channel selection
        self.setup_channel_selection(layout)
        
        # Filters
        self.setup_filters(layout)
        
        # Summary
        self.setup_summary(layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
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
        
        if 'channel_files' in self.header and self.header['channel_files']:
            channel_names = self.header['channel_files']
        else:
            channel_names = [f"Channel {i+1}" for i in range(n_channels)]
        
        for i in range(min(n_channels, 16)):
            name = channel_names[i] if i < len(channel_names) else f"Channel {i+1}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)
            self.channel_list.addItem(item)
            if i < 4:
                item.setSelected(True)
        
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
    
    def setup_filters(self, layout):
        """Setup filter controls"""
        group = QGroupBox("Filters")
        group_layout = QGridLayout(group)
        
        row = 0
        
        # Show notice if preprocessed
        if self.is_preprocessed:
            notice = QLabel("Preprocessed data - notch filter disabled")
            notice.setStyleSheet("color: #0078d4; font-weight: bold; padding: 5px;")
            group_layout.addWidget(notice, row, 0, 1, 2)
            row += 1
        
        # Notch filter - disabled for preprocessed data
        self.enable_notch = QCheckBox("50Hz Notch filter")
        if self.is_preprocessed:
            self.enable_notch.setChecked(False)
            self.enable_notch.setEnabled(False)
        else:
            self.enable_notch.setChecked(True)
        group_layout.addWidget(self.enable_notch, row, 0, 1, 2)
        row += 1
        
        # Bandpass filter
        self.enable_bandpass = QCheckBox("Bandpass filter")
        self.enable_bandpass.setChecked(True)
        group_layout.addWidget(self.enable_bandpass, row, 0, 1, 2)
        row += 1
        
        # Frequency settings
        group_layout.addWidget(QLabel("Low cutoff (Hz):"), row, 0)
        self.low_cutoff = QDoubleSpinBox()
        self.low_cutoff.setRange(0.1, 500.0)
        self.low_cutoff.setValue(80.0)
        self.low_cutoff.setSingleStep(10.0)
        group_layout.addWidget(self.low_cutoff, row, 1)
        row += 1
        
        group_layout.addWidget(QLabel("High cutoff (Hz):"), row, 0)
        self.high_cutoff = QDoubleSpinBox()
        self.high_cutoff.setRange(1.0, 1000.0)
        self.high_cutoff.setValue(250.0)
        self.high_cutoff.setSingleStep(10.0)
        group_layout.addWidget(self.high_cutoff, row, 1)
        row += 1
        
        # Presets
        preset_layout = QHBoxLayout()
        ripple_btn = QPushButton("Ripples (80-250Hz)")
        ripple_btn.clicked.connect(lambda: self.set_preset(80, 250))
        preset_layout.addWidget(ripple_btn)
        
        gamma_btn = QPushButton("Gamma (30-100Hz)")
        gamma_btn.clicked.connect(lambda: self.set_preset(30, 100))
        preset_layout.addWidget(gamma_btn)
        
        group_layout.addLayout(preset_layout, row, 0, 1, 2)
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
        self.enable_bandpass.toggled.connect(self.update_summary)
        self.enable_notch.toggled.connect(self.update_summary)
        self.low_cutoff.valueChanged.connect(self.update_summary)
        self.high_cutoff.valueChanged.connect(self.update_summary)
    
    def select_all_channels(self):
        for i in range(self.channel_list.count()):
            self.channel_list.item(i).setSelected(True)
    
    def clear_channels(self):
        self.channel_list.clearSelection()
    
    def set_preset(self, low, high):
        self.low_cutoff.setValue(low)
        self.high_cutoff.setValue(high)
        self.enable_bandpass.setChecked(True)
    
    def update_summary(self):
        """Update processing summary"""
        channels = self.get_selected_channels()
        
        if not channels:
            self.summary_label.setText("Please select at least one channel")
            return
        
        summary = [f"Channels: {len(channels)} selected"]
        summary.append(f"No resampling ({self.original_fs:.0f}Hz)")
        
        filters = []
        if self.enable_notch.isChecked():
            filters.append("50Hz notch")
        if self.enable_bandpass.isChecked():
            filters.append(f"{self.low_cutoff.value():.0f}-{self.high_cutoff.value():.0f}Hz bandpass")
        
        summary.append(f"Filters: {', '.join(filters)}" if filters else "No filters")
        
        if self.timestamps is not None:
            duration = self.timestamps[-1] - self.timestamps[0]
            summary.append(f"Duration: {duration:.1f}s (absolute timing preserved)")
        
        self.summary_label.setText("\n".join(summary))
    
    def get_selected_channels(self):
        """Get selected channel indices"""
        return [self.channel_list.item(i).data(Qt.UserRole) 
                for i in range(self.channel_list.count()) 
                if self.channel_list.item(i).isSelected()]
    
    def get_parameters(self):
        """Get processing parameters"""
        return {
            'channels': self.get_selected_channels(),
            'target_fs': self.original_fs,
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
        
        btn_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all)
        btn_layout.addWidget(select_all)
        
        clear_all = QPushButton("Clear")
        clear_all.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_all)
        layout.addLayout(btn_layout)
        
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


class DownsampleDialog(QDialog):
    """Downsampling dialog with smart frequency suggestions"""
    
    def __init__(self, data, header, timestamps, file_path, parent=None):
        super().__init__(parent)
        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.file_path = file_path
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
            info_text += f"Start time: {self.timestamps[0]:.3f} seconds"
        else:
            duration = len(self.data) / self.original_fs
            info_text += f"Duration: {duration:.1f} seconds"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                padding: 10px; background: #f8f9fa;
                border: 1px solid #dee2e6; border-radius: 4px;
                font-family: monospace;
            }
        """)
        layout.addWidget(info_label)
        
        # Downsample section
        downsample_group = QGroupBox("Downsample Settings")
        downsample_layout = QVBoxLayout(downsample_group)
        
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
        
        suggestions = self.get_smart_suggestions()
        self.freq_combo.addItems([str(f) for f in suggestions])
        
        default_freq = 1000 if 1000 in suggestions else suggestions[0]
        self.freq_combo.setCurrentText(str(default_freq))
        
        freq_layout.addWidget(self.freq_combo)
        downsample_layout.addLayout(freq_layout)
        layout.addWidget(downsample_group)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("""
            QLabel {
                padding: 8px; background: #e9ecef;
                border: 1px solid #ced4da; border-radius: 3px;
                font-family: monospace; font-size: 9pt;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.freq_combo.currentTextChanged.connect(self.update_preview)
        self.update_preview()
    
    def get_smart_suggestions(self):
        """Generate smart frequency suggestions based on original rate"""
        suggestions = []
        common_rates = [500, 1000, 2000, 5000, 10000, 20000]
        
        for rate in common_rates:
            if rate < self.original_fs * 0.9:
                suggestions.append(rate)
        
        for frac in [0.1, 0.2, 0.25, 0.5]:
            suggested = int(self.original_fs * frac)
            if suggested >= 500 and suggested not in suggestions:
                suggestions.append(suggested)
        
        suggestions = sorted(set(suggestions))
        
        if len(suggestions) < 3:
            for rate in [500, 1000, 2000]:
                if rate not in suggestions and rate < self.original_fs:
                    suggestions.append(rate)
        
        return sorted(set(suggestions))
    
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
            
            downsample_factor = self.original_fs / target_fs
            new_samples = int(len(self.data) / downsample_factor)
            
            if self.timestamps is not None:
                duration = self.timestamps[-1] - self.timestamps[0]
            else:
                duration = len(self.data) / self.original_fs
            
            nyquist_new = target_fs / 2
            
            preview_text = f"Downsample: {self.original_fs:.0f}Hz -> {target_fs:.0f}Hz\n"
            preview_text += f"Factor: {downsample_factor:.1f}x reduction\n"
            preview_text += f"Samples: {len(self.data):,} -> {new_samples:,}\n"
            preview_text += f"Duration: {duration:.1f}s (preserved)\n"
            preview_text += f"Nyquist: {nyquist_new:.0f}Hz\n"
            
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