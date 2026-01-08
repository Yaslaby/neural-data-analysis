import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QFileDialog, QTabWidget, QTextEdit,
                             QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QComboBox, QGroupBox, QProgressBar, QSplitter,
                             QListWidget, QListWidgetItem, QMessageBox, QStatusBar, 
                             QMenuBar, QAction, QToolBar, QScrollArea, QDialog, 
                             QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
                             QFrame, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, spectrogram
import struct
import warnings
warnings.filterwarnings('ignore')

def read_continuous_file(filepath):
    """Read Open Ephys .continuous file"""
    try:
        with open(filepath, 'rb') as f:
            # Read header
            header = f.read(1024).decode('ascii', errors='ignore')
            
            # Extract info from header
            lines = header.split('\n')
            fs = 30000  # Default sampling rate
            channel = 'CH1'  # Default channel
            
            for line in lines:
                if 'sampleRate' in line:
                    fs = float(line.split('=')[1].strip())
                elif 'channel' in line:
                    channel = line.split('=')[1].strip()
            
            # Read data
            f.seek(1024)  # Skip header
            data = []
            
            while True:
                # Read record
                timestamp_bytes = f.read(8)
                if len(timestamp_bytes) < 8:
                    break
                    
                n_samples_bytes = f.read(2)
                if len(n_samples_bytes) < 2:
                    break
                
                n_samples = struct.unpack('<H', n_samples_bytes)[0]
                
                # Read samples (16-bit signed integers)
                samples_bytes = f.read(n_samples * 2)
                if len(samples_bytes) < n_samples * 2:
                    break
                
                samples = struct.unpack('<' + 'h' * n_samples, samples_bytes)
                data.extend(samples)
                
                # Skip record marker
                f.read(10)
            
            # Convert to numpy array and microvolts
            data = np.array(data, dtype=np.float32) * 0.195  # Convert to microvolts
            
            return data, fs, channel
            
    except Exception as e:
        raise Exception(f"Error reading .continuous file: {str(e)}")

class DataSet:
    """Class representing a neural data set"""
    
    def __init__(self, name, data, fs, channel_names=None, filepath=None):
        self.name = name
        self.data = data
        self.fs = fs
        self.channel_names = channel_names or [f"CH{i+1}" for i in range(data.shape[1] if data.ndim > 1 else 1)]
        self.filepath = filepath
        self.annotations = []
        self.events = []
        self.data_type = "Raw"
        
        # Ensure data is 2D
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
    
    @property
    def n_samples(self):
        return len(self.data)
    
    @property
    def n_channels(self):
        return self.data.shape[1]
    
    @property
    def duration(self):
        return self.n_samples / self.fs
    
    def get_info_dict(self):
        """Get information dictionary for display"""
        return {
            "Data type": self.data_type,
            "Channels": f"{self.n_channels}",
            "Samples": f"{self.n_samples:,}",
            "Length": f"{self.duration:.2f} s",
            "Sampling frequency": f"{self.fs:.1f} Hz",
            "Size in memory": f"{self.data.nbytes / 1024 / 1024:.1f} MB",
            "Annotations": f"{len(self.annotations)}",
            "Events": f"{len(self.events)}"
        }

class ConcatenateDialog(QDialog):
    """Dialog for concatenating data sets"""
    
    def __init__(self, datasets, current_dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Append data")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        label = QLabel("Select data sets to append to the current data set:")
        layout.addWidget(label)
        
        # Source and destination panels
        content_layout = QHBoxLayout()
        
        # Source panel
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        self.source_list = QListWidget()
        self.source_list.setSelectionMode(QListWidget.MultiSelection)
        
        for dataset in datasets:
            if dataset.name != current_dataset.name:
                item = QListWidgetItem(dataset.name)
                item.setData(Qt.UserRole, dataset)
                self.source_list.addItem(item)
        
        source_layout.addWidget(self.source_list)
        content_layout.addWidget(source_group)
        
        # Arrow buttons
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        self.add_button = QPushButton("→")
        self.add_button.clicked.connect(self.add_datasets)
        button_layout.addWidget(self.add_button)
        button_layout.addStretch()
        content_layout.addLayout(button_layout)
        
        # Destination panel
        dest_group = QGroupBox("Destination")
        dest_layout = QVBoxLayout(dest_group)
        self.dest_list = QListWidget()
        dest_layout.addWidget(self.dest_list)
        content_layout.addWidget(dest_group)
        
        layout.addLayout(content_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def add_datasets(self):
        """Add selected datasets to destination"""
        for item in self.source_list.selectedItems():
            new_item = QListWidgetItem(item.text())
            new_item.setData(Qt.UserRole, item.data(Qt.UserRole))
            self.dest_list.addItem(new_item)
    
    def get_selected_datasets(self):
        """Get selected datasets for concatenation"""
        datasets = []
        for i in range(self.dest_list.count()):
            item = self.dest_list.item(i)
            datasets.append(item.data(Qt.UserRole))
        return datasets

class RippleDetectionDialog(QDialog):
    """Dialog for ripple detection parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detect Ripples")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Frequency band
        freq_group = QGroupBox("Ripple Frequency Band")
        freq_layout = QGridLayout(freq_group)
        
        freq_layout.addWidget(QLabel("Low frequency (Hz):"), 0, 0)
        self.low_freq = QSpinBox()
        self.low_freq.setRange(50, 200)
        self.low_freq.setValue(80)
        freq_layout.addWidget(self.low_freq, 0, 1)
        
        freq_layout.addWidget(QLabel("High frequency (Hz):"), 1, 0)
        self.high_freq = QSpinBox()
        self.high_freq.setRange(100, 500)
        self.high_freq.setValue(250)
        freq_layout.addWidget(self.high_freq, 1, 1)
        
        layout.addWidget(freq_group)
        
        # Detection parameters
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QGridLayout(detection_group)
        
        detection_layout.addWidget(QLabel("Threshold (SD):"), 0, 0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1.0, 10.0)
        self.threshold.setValue(3.0)
        self.threshold.setSingleStep(0.1)
        detection_layout.addWidget(self.threshold, 0, 1)
        
        detection_layout.addWidget(QLabel("Min duration (ms):"), 1, 0)
        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(5.0, 100.0)
        self.min_duration.setValue(15.0)
        detection_layout.addWidget(self.min_duration, 1, 1)
        
        detection_layout.addWidget(QLabel("Max duration (ms):"), 2, 0)
        self.max_duration = QDoubleSpinBox()
        self.max_duration.setRange(50.0, 500.0)
        self.max_duration.setValue(200.0)
        detection_layout.addWidget(self.max_duration, 2, 1)
        
        layout.addWidget(detection_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

class AnalysisWorker(QThread):
    """Worker thread for ripple detection"""
    
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data, fs, params):
        super().__init__()
        self.data = data
        self.fs = fs
        self.params = params
        
    def run(self):
        try:
            result = self.detect_ripples()
            self.analysis_completed.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def detect_ripples(self):
        """Detect ripple events in the signal"""
        # Filter signal in ripple band
        nyquist = self.fs / 2
        low_freq = self.params['low_freq'] / nyquist
        high_freq = self.params['high_freq'] / nyquist
        
        b, a = butter(4, [low_freq, high_freq], btype='band')
        filtered_signal = filtfilt(b, a, self.data)
        self.progress_updated.emit(30)
        
        # Compute envelope using Hilbert transform
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        self.progress_updated.emit(60)
        
        # Smooth envelope
        window_size = int(self.fs * 0.01)  # 10ms window
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Detect ripples using threshold
        threshold = self.params['threshold']
        mean_env = np.mean(envelope_smooth)
        std_env = np.std(envelope_smooth)
        ripple_threshold = mean_env + threshold * std_env
        
        # Find ripple events
        above_threshold = envelope_smooth > ripple_threshold
        ripple_events = []
        
        in_ripple = False
        start_idx = 0
        min_duration = int(self.fs * self.params['min_duration'] / 1000.0)
        max_duration = int(self.fs * self.params['max_duration'] / 1000.0)
        
        for i, above in enumerate(above_threshold):
            if above and not in_ripple:
                in_ripple = True
                start_idx = i
            elif not above and in_ripple:
                in_ripple = False
                duration_samples = i - start_idx
                if min_duration <= duration_samples <= max_duration:
                    ripple_events.append({
                        'start': start_idx,
                        'end': i,
                        'duration': duration_samples / self.fs,
                        'peak_amplitude': np.max(envelope_smooth[start_idx:i]),
                        'time': start_idx / self.fs
                    })
        
        self.progress_updated.emit(100)
        
        return {
            'filtered_signal': filtered_signal,
            'envelope': envelope_smooth,
            'ripple_events': ripple_events,
            'threshold': ripple_threshold,
            'detection_params': self.params
        }

class PlotCanvas(FigureCanvas):
    """Main plotting canvas"""
    
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('white')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Set white background
        self.setStyleSheet("background-color: white;")
        
    def plot_data(self, dataset, time_range=None, channels=None):
        """Plot neural data"""
        self.fig.clear()
        
        if dataset is None:
            return
        
        # Select channels to plot
        if channels is None:
            channels = list(range(min(10, dataset.n_channels)))  # Max 10 channels
        
        # Select time range
        if time_range is None:
            start_sample, end_sample = 0, min(int(10 * dataset.fs), dataset.n_samples)
        else:
            start_sample = int(time_range[0] * dataset.fs)
            end_sample = int(time_range[1] * dataset.fs)
        
        # Create time axis
        times = np.arange(start_sample, end_sample) / dataset.fs
        
        # Plot channels
        n_channels = len(channels)
        if n_channels == 0:
            return
        
        for i, ch_idx in enumerate(channels):
            ax = self.fig.add_subplot(n_channels, 1, i + 1)
            
            if ch_idx < dataset.n_channels:
                signal_data = dataset.data[start_sample:end_sample, ch_idx]
                ax.plot(times, signal_data, 'k-', linewidth=0.8)
                ax.set_ylabel(f'{dataset.channel_names[ch_idx]}\n(μV)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Remove x-axis labels except for last subplot
                if i < n_channels - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (s)', fontsize=12)
        
        self.fig.suptitle(f'{dataset.name} - Neural Data', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.draw()

class OpenEphysLab(QMainWindow):
    """Main application window - MNELAB style for Open Ephys data"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenEphysLab - Neural Data Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data management
        self.datasets = []
        self.current_dataset = None
        
        # UI setup
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_status_bar()
        self.connect_signals()
        
        # Apply MNELAB-style black and white theme
        self.apply_mnelab_style()
        
        # Show welcome message
        self.show_welcome_dialog()
    
    def apply_mnelab_style(self):
        """Apply MNELAB-inspired black and white styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                color: black;
            }
            QMenuBar {
                background-color: white;
                color: black;
                border-bottom: 1px solid #d0d0d0;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }
            QToolBar {
                background-color: white;
                border: 1px solid #d0d0d0;
                color: black;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                color: black;
                selection-background-color: #0078d4;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                color: black;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                margin-top: 10px;
                padding-top: 5px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: white;
                border: 1px solid #d0d0d0;
                padding: 6px 12px;
                color: black;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
            }
            QStatusBar {
                background-color: #f0f0f0;
                border-top: 1px solid #d0d0d0;
            }
        """)
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left sidebar for data sets
        self.setup_sidebar(main_layout)
        
        # Main content area
        self.setup_main_content(main_layout)
        
        # Right info panel
        self.setup_info_panel(main_layout)
    
    def setup_sidebar(self, parent_layout):
        """Setup left sidebar with data sets"""
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # Data sets list
        datasets_label = QLabel("Data sets")
        datasets_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(datasets_label)
        
        self.datasets_list = QListWidget()
        self.datasets_list.setMaximumHeight(300)
        sidebar_layout.addWidget(self.datasets_list)
        
        # Add stretch to push everything to top
        sidebar_layout.addStretch()
        
        parent_layout.addWidget(sidebar_widget)
    
    def setup_main_content(self, parent_layout):
        """Setup main content area with tabs"""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Plot tab
        self.plot_canvas = PlotCanvas(width=10, height=6)
        plot_scroll = QScrollArea()
        plot_scroll.setWidget(self.plot_canvas)
        plot_scroll.setWidgetResizable(True)
        self.tab_widget.addTab(plot_scroll, "Plot")
        
        content_layout.addWidget(self.tab_widget)
        parent_layout.addWidget(content_widget)
    
    def setup_info_panel(self, parent_layout):
        """Setup right info panel"""
        info_widget = QWidget()
        info_widget.setFixedWidth(250)
        info_layout = QVBoxLayout(info_widget)
        
        # Info label
        info_label = QLabel("Info")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        info_layout.addWidget(info_label)
        
        # Info table
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.setAlternatingRowColors(True)
        info_layout.addWidget(self.info_table)
        
        parent_layout.addWidget(info_widget)
    
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        close_action = QAction('Close', self)
        close_action.triggered.connect(self.close_dataset)
        file_menu.addAction(close_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        append_action = QAction('Append data...', self)
        append_action.triggered.connect(self.append_data)
        edit_menu.addAction(append_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        ripple_action = QAction('Detect Ripples...', self)
        ripple_action.triggered.connect(self.detect_ripples)
        tools_menu.addAction(ripple_action)
        
        # Plot menu
        plot_menu = menubar.addMenu('Plot')
        
        plot_data_action = QAction('Plot data', self)
        plot_data_action.triggered.connect(self.plot_data)
        plot_menu.addAction(plot_data_action)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = self.addToolBar('Main')
        
        # Open files
        open_action = QAction('📁', self)
        open_action.setToolTip('Open files')
        open_action.triggered.connect(self.open_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Plot data
        plot_action = QAction('📊', self)
        plot_action.setToolTip('Plot data')
        plot_action.triggered.connect(self.plot_data)
        toolbar.addAction(plot_action)
        
        # Detect ripples
        ripple_action = QAction('🔬', self)
        ripple_action.setToolTip('Detect ripples')
        ripple_action.triggered.connect(self.detect_ripples)
        toolbar.addAction(ripple_action)
        
        # Progress bar
        toolbar.addSeparator()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        toolbar.addWidget(self.progress_bar)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Import Open Ephys .continuous files to begin")
    
    def connect_signals(self):
        """Connect UI signals"""
        self.datasets_list.currentItemChanged.connect(self.dataset_changed)
        self.datasets_list.itemChanged.connect(self.dataset_renamed)
    
    def show_welcome_dialog(self):
        """Show welcome dialog"""
        msg = QMessageBox(self)
        msg.setWindowTitle("OpenEphysLab")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Welcome to OpenEphysLab")
        msg.setInformativeText(
            "Import Open Ephys Data (.continuous files)\n\n"
            "• Select File → Open... to import .continuous files\n"
            "• Multiple channels can be loaded and concatenated\n"
            "• Use Tools → Detect Ripples for sharp-wave ripple analysis\n"
            "• Plot data to visualize neural signals\n\n"
            "This interface is designed for hippocampal ripple detection\n"
            "from Open Ephys recordings."
        )
        msg.exec_()
    
    def open_files(self):
        """Open Open Ephys .continuous files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Open Ephys Data", "", 
            "Open Ephys files (*.continuous);;All Files (*)"
        )
        
        if file_paths:
            for file_path in file_paths:
                try:
                    # Read .continuous file
                    data, fs, channel = read_continuous_file(file_path)
                    
                    # Create dataset
                    filename = os.path.basename(file_path)
                    name = filename.replace('.continuous', '')
                    
                    dataset = DataSet(name, data, fs, [channel], file_path)
                    self.datasets.append(dataset)
                    
                    # Add to UI
                    item = QListWidgetItem(name)
                    item.setData(Qt.UserRole, dataset)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    self.datasets_list.addItem(item)
                    
                    # Set as current
                    self.datasets_list.setCurrentItem(item)
                    
                    self.status_bar.showMessage(f"Loaded: {filename}")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Import Error", 
                                       f"Failed to import {file_path}:\n{str(e)}")
    
    def dataset_changed(self, current, previous):
        """Handle dataset selection change"""
        if current:
            self.current_dataset = current.data(Qt.UserRole)
            self.update_info_panel()
            self.plot_data()
        else:
            self.current_dataset = None
            self.clear_info_panel()
    
    def dataset_renamed(self, item):
        """Handle dataset rename"""
        dataset = item.data(Qt.UserRole)
        dataset.name = item.text()
    
    def update_info_panel(self):
        """Update info panel with current dataset info"""
        if not self.current_dataset:
            return
        
        info_dict = self.current_dataset.get_info_dict()
        
        self.info_table.setRowCount(len(info_dict))
        for i, (key, value) in enumerate(info_dict.items()):
            self.info_table.setItem(i, 0, QTableWidgetItem(key))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.info_table.resizeColumnsToContents()
    
    def clear_info_panel(self):
        """Clear info panel"""
        self.info_table.setRowCount(0)
    
    def plot_data(self):
        """Plot current dataset"""
        if self.current_dataset:
            self.plot_canvas.plot_data(self.current_dataset)
    
    def append_data(self):
        """Append data sets (concatenate)"""
        if not self.current_dataset:
            QMessageBox.warning(self, "No Data", "Please select a data set first.")
            return
        
        if len(self.datasets) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 data sets to concatenate.")
            return
        
        dialog = ConcatenateDialog(self.datasets, self.current_dataset, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_datasets = dialog.get_selected_datasets()
            if selected_datasets:
                try:
                    # Concatenate data
                    all_data = [self.current_dataset.data]
                    for dataset in selected_datasets:
                        all_data.append(dataset.data)
                    
                    concatenated_data = np.vstack(all_data)
                    
                    # Create new dataset
                    new_name = f"{self.current_dataset.name} (appended)"
                    new_dataset = DataSet(
                        new_name, 
                        concatenated_data, 
                        self.current_dataset.fs,
                        self.current_dataset.channel_names
                    )
                    
                    self.datasets.append(new_dataset)
                    
                    # Add to UI
                    item = QListWidgetItem(new_name)
                    item.setData(Qt.UserRole, new_dataset)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    self.datasets_list.addItem(item)
                    self.datasets_list.setCurrentItem(item)
                    
                    self.status_bar.showMessage("Data sets concatenated successfully")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Concatenation Error", 
                                       f"Failed to concatenate data:\n{str(e)}")
    
    def close_dataset(self):
        """Close current dataset"""
        if not self.current_dataset:
            return
        
        # Remove from list
        current_item = self.datasets_list.currentItem()
        if current_item:
            row = self.datasets_list.row(current_item)
            self.datasets_list.takeItem(row)
            
            # Remove from datasets
            self.datasets.remove(self.current_dataset)
            
            # Clear current dataset
            self.current_dataset = None
            self.clear_info_panel()
            self.plot_canvas.fig.clear()
            self.plot_canvas.draw()
            
            self.status_bar.showMessage("Dataset closed")
    
    def detect_ripples(self):
        """Detect ripples in current dataset"""
        if not self.current_dataset:
            QMessageBox.warning(self, "No Data", "Please select a data set first.")
            return
        
        dialog = RippleDetectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Get parameters
            params = {
                'low_freq': dialog.low_freq.value(),
                'high_freq': dialog.high_freq.value(),
                'threshold': dialog.threshold.value(),
                'min_duration': dialog.min_duration.value(),
                'max_duration': dialog.max_duration.value()
            }
            
            # Use first channel for ripple detection
            channel_data = self.current_dataset.data[:, 0]
            
            # Start analysis
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Detecting ripples...")
            
            self.worker = AnalysisWorker(channel_data, self.current_dataset.fs, params)
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            self.worker.analysis_completed.connect(self.on_ripple_detection_completed)
            self.worker.error_occurred.connect(self.on_analysis_error)
            self.worker.start()
    
    def on_ripple_detection_completed(self, results):
        """Handle ripple detection completion"""
        self.progress_bar.setVisible(False)
        
        n_ripples = len(results['ripple_events'])
        
        if n_ripples > 0:
            # Add ripples as annotations
            for event in results['ripple_events']:
                annotation = {
                    'time': event['time'],
                    'duration': event['duration'],
                    'description': f"Ripple ({event['duration']*1000:.1f}ms)"
                }
                self.current_dataset.annotations.append(annotation)
            
            # Update info panel
            self.update_info_panel()
            
            # Show results
            durations = [event['duration'] for event in results['ripple_events']]
            amplitudes = [event['peak_amplitude'] for event in results['ripple_events']]
            
            summary = f"""Ripple Detection Results:

Total ripples detected: {n_ripples}
Average duration: {np.mean(durations)*1000:.1f} ± {np.std(durations)*1000:.1f} ms
Average amplitude: {np.mean(amplitudes):.2f} ± {np.std(amplitudes):.2f} μV
Ripple rate: {n_ripples/self.current_dataset.duration:.2f} ripples/second

Detection parameters:
• Frequency band: {results['detection_params']['low_freq']}-{results['detection_params']['high_freq']} Hz
• Threshold: {results['detection_params']['threshold']} SD
• Duration range: {results['detection_params']['min_duration']}-{results['detection_params']['max_duration']} ms"""
            
            QMessageBox.information(self, "Ripple Detection Results", summary)
            
            # Plot results
            self.plot_ripple_results(results)
            
        else:
            QMessageBox.information(self, "Ripple Detection Results", 
                                  "No ripples detected with current parameters.\n\n"
                                  "Try adjusting:\n"
                                  "• Lower threshold\n"
                                  "• Different frequency band\n"
                                  "• Different duration limits")
        
        self.status_bar.showMessage(f"Ripple detection completed - {n_ripples} ripples found")
    
    def plot_ripple_results(self, results):
        """Plot ripple detection results"""
        self.plot_canvas.fig.clear()
        
        # Create subplots
        ax1 = self.plot_canvas.fig.add_subplot(3, 1, 1)
        ax2 = self.plot_canvas.fig.add_subplot(3, 1, 2)
        ax3 = self.plot_canvas.fig.add_subplot(3, 1, 3)
        
        # Time axis (show first 10 seconds)
        duration_to_show = min(10.0, self.current_dataset.duration)
        samples_to_show = int(duration_to_show * self.current_dataset.fs)
        times = np.arange(samples_to_show) / self.current_dataset.fs
        
        # Original signal
        original_data = self.current_dataset.data[:samples_to_show, 0]
        ax1.plot(times, original_data, 'k-', linewidth=0.8)
        ax1.set_ylabel('Raw Signal\n(μV)', fontsize=10)
        ax1.set_title(f'{self.current_dataset.name} - Ripple Detection Results', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Filtered signal
        filtered_data = results['filtered_signal'][:samples_to_show]
        ax2.plot(times, filtered_data, 'b-', linewidth=0.8)
        ax2.set_ylabel(f'Filtered\n({results["detection_params"]["low_freq"]}-{results["detection_params"]["high_freq"]} Hz)\n(μV)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Envelope and threshold
        envelope_data = results['envelope'][:samples_to_show]
        ax3.plot(times, envelope_data, 'g-', linewidth=1.0, label='Envelope')
        ax3.axhline(y=results['threshold'], color='r', linestyle='--', linewidth=1.5, label='Threshold')
        ax3.set_ylabel('Envelope\n(μV)', fontsize=10)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mark detected ripples
        for event in results['ripple_events']:
            if event['time'] < duration_to_show:
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(event['time'], color='red', alpha=0.7, linewidth=2)
                    ax.axvspan(event['time'], event['time'] + event['duration'], 
                             color='red', alpha=0.2)
        
        # Set same x-limits for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, duration_to_show)
        
        self.plot_canvas.fig.tight_layout()
        self.plot_canvas.draw()
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", f"Ripple detection failed:\n{error_msg}")
        self.status_bar.showMessage("Analysis failed")

def create_sample_continuous_file(filepath, duration=30, fs=30000):
    """Create a sample .continuous file for testing"""
    try:
        # Generate sample neural data
        t = np.arange(0, duration, 1/fs)
        
        # Base signal with theta and gamma
        signal_data = (
            np.random.normal(0, 50, len(t)) +  # Noise
            100 * np.sin(2 * np.pi * 8 * t) +  # Theta
            30 * np.sin(2 * np.pi * 40 * t)    # Gamma
        )
        
        # Add some ripple events
        ripple_times = np.random.uniform(5, duration-5, 8)
        for ripple_time in ripple_times:
            start_idx = int(ripple_time * fs)
            ripple_duration = int(np.random.uniform(0.05, 0.15) * fs)
            ripple_freq = np.random.uniform(150, 200)
            
            ripple_t = np.arange(ripple_duration) / fs
            envelope = np.exp(-((ripple_t - ripple_t[len(ripple_t)//2]) / 0.03)**2)
            ripple = envelope * 200 * np.sin(2 * np.pi * ripple_freq * ripple_t)
            
            end_idx = min(start_idx + ripple_duration, len(signal_data))
            signal_data[start_idx:end_idx] += ripple[:end_idx-start_idx]
        
        # Convert to 16-bit integers
        signal_data = signal_data.astype(np.int16)
        
        # Create .continuous file
        with open(filepath, 'wb') as f:
            # Write header (1024 bytes)
            header = f"header.format = 'Open Ephys Data Format';\n"
            header += f"header.version = '0.4';\n"
            header += f"header.header_bytes = 1024;\n"
            header += f"header.description = 'Sample data';\n"
            header += f"header.date_created = '2024-01-01';\n"
            header += f"header.channel = 'CH1';\n"
            header += f"header.channelType = 'Continuous';\n"
            header += f"header.sampleRate = {fs};\n"
            header += f"header.blockLength = 1024;\n"
            header += f"header.bufferSize = 1024;\n"
            header += f"header.bitVolts = 0.195;\n"
            
            # Pad header to 1024 bytes
            header = header.ljust(1024, '\x00')
            f.write(header.encode('ascii'))
            
            # Write data in blocks
            block_size = 1024
            for i in range(0, len(signal_data), block_size):
                # Timestamp (8 bytes)
                timestamp = i
                f.write(struct.pack('<Q', timestamp))
                
                # Number of samples (2 bytes)
                n_samples = min(block_size, len(signal_data) - i)
                f.write(struct.pack('<H', n_samples))
                
                # Sample data
                block_data = signal_data[i:i+n_samples]
                for sample in block_data:
                    f.write(struct.pack('<h', sample))
                
                # Record marker (10 bytes)
                f.write(b'\x00' * 10)
        
        return True
        
    except Exception as e:
        print(f"Error creating sample file: {e}")
        return False

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("OpenEphysLab")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Neural Data Analysis Lab")
    
    # Create and show main window
    window = OpenEphysLab()
    window.show()
    
    # Ask if user wants sample data
    reply = QMessageBox.question(
        window, 'Sample Data', 
        'Would you like to create sample Open Ephys .continuous files for testing?\n\n'
        'This will create realistic hippocampal data with:\n'
        '• Theta and gamma oscillations\n'
        '• Embedded sharp-wave ripple events\n'
        '• Proper Open Ephys file format\n\n'
        'Perfect for testing ripple detection!',
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        try:
            # Create sample files
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            sample_files = []
            for i, session in enumerate(['001', '002', '003']):
                filepath = os.path.join(temp_dir, f'100_CH{i+1}_{session}.continuous')
                if create_sample_continuous_file(filepath, duration=20):
                    sample_files.append(filepath)
            
            if sample_files:
                info_msg = f"""Sample .continuous files created successfully!

Files created:
{chr(10).join(['• ' + os.path.basename(f) for f in sample_files])}

Location: {temp_dir}

Now try:
1. File → Open... to import these files
2. Select multiple files to see them in the sidebar
3. Use Edit → Append data... to concatenate sessions
4. Tools → Detect Ripples to find sharp-wave ripples
5. View results in the plot area

The files contain realistic hippocampal data with embedded ripples!"""
                
                QMessageBox.information(window, "Sample Files Created", info_msg)
                
                # Auto-load the first file
                window.open_files()
                
            else:
                QMessageBox.warning(window, "Error", "Failed to create sample files")
                
        except Exception as e:
            QMessageBox.warning(window, "Error", f"Failed to create sample data:\n{str(e)}")
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()