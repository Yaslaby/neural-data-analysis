import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QSplitter, QListWidget, QListWidgetItem,
                             QTableWidget, QTableWidgetItem, QStackedWidget, 
                             QMessageBox, QFileDialog, QDialog, QProgressDialog,
                             QToolBar, QSlider, QComboBox, QPushButton, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QMutex
from PyQt5.QtGui import QFont

# Import existing modules
from PlotManager import PlotManager
from integrated_mne_processing import process_for_ripples_mne_standard as process_for_ripples
from MultiChannelLoader import MultiChannelLoader, load_single_channel
from annotation import (AnnotationManager, AnnotationWidget, add_right_click_annotation, AnnotationControls,  add_drag_to_mark_annotation)
from signal_processing import SignalProcessingWorker
from dialogs import (FilterDialog, ChannelSelectionDialog, RippleDetectionDialog, 
                     ExportDialog, ProgressDialog, PreprocessingDialog, DownsampleDialog)

# Import the new combined dialog
from dialogs import MultiChannelDownsampleDialog
from sleep_scoring_dialog import SleepScoringDialog
from ripple_detection import find_ripples_karlsson
import pyqtgraph as pg
import numpy as np
import traceback
from scipy import signal


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
        self.worker = None
        self.worker_thread = None
        self.progress_dialog = None
        QApplication.processEvents()
    
    def set_data(self, data, target_fs, original_fs, original_timestamps):
        self.data = data
        self.target_fs = target_fs
        self.original_fs = original_fs
        self.original_timestamps = original_timestamps
    
    def run(self):
        try:
            import mne
            
            self.progress_updated.emit(20, "Creating MNE object...")
            
            # Handle multichannel data
            if self.data.ndim == 1:
                n_channels = 1
                data_for_mne = self.data.reshape(1, -1)
                info = mne.create_info(['CH1'], self.original_fs, ['seeg'])
            else:
                n_channels = self.data.shape[1]
                data_for_mne = self.data.T  # MNE expects channels x samples
                ch_names = [f'CH{i+1}' for i in range(n_channels)]
                info = mne.create_info(ch_names, self.original_fs, ['seeg'] * n_channels)
            
            raw = mne.io.RawArray(data_for_mne, info, verbose=False)
            
            self.progress_updated.emit(60, "Downsampling...")
            raw_downsampled = raw.copy().resample(self.target_fs, verbose=False)
            downsampled_data = raw_downsampled.get_data().T  # Convert back to samples x channels
            
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
            
            # Get raw data
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
            # Process each channel
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
            
            # Combine processed channels
            if len(processed_channels) > 1:
                proc_data = np.column_stack(processed_channels)
            else:
                proc_data = processed_channels[0].reshape(-1, 1)
            
            # Create timestamps for processed data (preserving absolute timing)
            if self.timestamps is not None:
                start_time = self.timestamps[0]
                end_time = self.timestamps[-1]
                proc_timestamps = np.linspace(start_time, end_time, len(proc_data))
            else:
                proc_timestamps = np.arange(len(proc_data)) / target_fs
            
            # Get channel names
            if 'channel_files' in self.header and self.header['channel_files']:
                channel_names = [self.header['channel_files'][i] for i in channels]
            else:
                channel_names = [f"CH{i+1}" for i in channels]
            
            self.progress_updated.emit(100, "Processing complete!")
            
            # Emit results
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
            
            # Create MNE raw object
            info = mne.create_info([f'CH{channel_idx+1}'], original_fs, ['seeg'])
            raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
            
            # Resample if needed (this should be minimal since we're already downsampled)
            if target_fs != original_fs:
                raw = raw.copy().resample(target_fs, verbose=False)
            
            # Apply notch filter if enabled
            if self.params['notch_enabled']:
                raw._data[0] = notch_filter(
                    raw._data[0], target_fs, freqs=[50], 
                    method='fir', phase='zero', verbose=False
                )
            
            # Apply bandpass filter if enabled
            if self.params['bandpass_enabled']:
                raw = raw.copy().filter(
                    l_freq=self.params['low_cutoff'],
                    h_freq=self.params['high_cutoff'],
                    method='fir', phase='zero', verbose=False
                )
            
            return raw._data[0]
            
        except Exception as e:
            raise Exception(f"MNE processing failed: {str(e)}")

class OpenEphysMainWindow(QMainWindow):
    """Main application window with complete downsample + preprocess workflow"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data storage
        self.current_data = None
        self.current_header = None
        self.datasets = []
        self.filtered_datasets = []
        self.current_index = -1
        
        # Threading objects
        self.worker = None
        self.worker_thread = None
        self.progress_dialog = None
        
        # Create UI
        self.create_mnelab_style_ui()
        self.setup_managers()
        self.connect_signals()
        self.apply_mnelab_styling()
        
        # Set window properties
        self.setWindowTitle("Neural Data Analysis")
        self.setMinimumSize(1200, 800)
        
        #sleep scoring dialog
        self.sleep_scoring_data = None
        self.sleep_scoring_items = [] 

        print("OpenEphysLab initialized successfully!")
    
    def create_mnelab_style_ui(self):
        """Create MNELAB-style UI with sidebar and main area"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        self.create_sidebar()
        self.create_main_content()
        self.create_info_panel()
        
        self.splitter.setSizes([200, 800, 250])
        self.create_menus()
        self.sidebar_widget.hide()
        self.splitter.widget(2).hide()  # Hide info panel initially
    
    def create_sidebar(self):
        """Create left sidebar with datasets and filtered data lists"""
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self.sidebar_widget)
    
        # Data sets section
        datasets_label = QLabel("Data sets")
        datasets_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(datasets_label)
        
        self.datasets_list = QListWidget()
        self.datasets_list.setMaximumHeight(300)
        sidebar_layout.addWidget(self.datasets_list)
        
        # Filtered Data section
        filtered_label = QLabel("Processed Data")
        filtered_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(filtered_label)
        
        self.filtered_list = QListWidget()
        self.filtered_list.setMaximumHeight(300)
        sidebar_layout.addWidget(self.filtered_list)
        
        sidebar_layout.addStretch()
        self.splitter.addWidget(self.sidebar_widget)
    
    def create_main_content(self):
        """Create main content area"""
        self.content_stack = QStackedWidget()
        
        # Empty page
        self.empty_page = self.create_empty_page()
        self.content_stack.addWidget(self.empty_page)
        
        # Plot page
        self.plot_page = self.create_plot_page()
        self.content_stack.addWidget(self.plot_page)
        
        self.content_stack.setCurrentWidget(self.empty_page)
        self.splitter.addWidget(self.content_stack)
    
    def create_empty_page(self):
        """Create empty welcome page"""
        empty_widget = QWidget()
        layout = QVBoxLayout(empty_widget)
        layout.addStretch()
        
        welcome_label = QLabel(
            "Welcome to Neural Data Analysis\n\n"
            "Load Open Ephys .continuous files to begin neural data analysis.\n\n"
            "Workflow:\n"
            "1. File → Open: Load & downsample file\n"
            "2. Edit → Preprocess: Apply filters with comparison view\n"
            "Features:\n"
            "• Multi-channel visualization\n" 
            "• Manual annotation system\n"
            "• Before/after comparison plots"
        )
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setFont(QFont("Arial", 12))
        welcome_label.setStyleSheet("color: #666; padding: 20px;")
        layout.addWidget(welcome_label)
        layout.addStretch()
        
        return empty_widget
    
    def create_plot_page(self):
        """Create plot page with visualization and annotations + rezizable panels"""
        plot_widget = QWidget()
        main_layout = QVBoxLayout(plot_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create vertical splitter for resizable plot and annotation panels
        self.plot_splitter = QSplitter(Qt.Vertical)
        self.plot_splitter.setHandleWidth(5)  # Make handle visible
        self.plot_splitter.setChildrenCollapsible(False)
        
        # Create container for plot + slider (top section of splitter)
        self.plot_container = QWidget()
        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container_layout.setSpacing(0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='µV')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add plot to container
        self.plot_container_layout.addWidget(self.plot_widget)
        
        # Add plot container to splitter FIRST
        self.plot_splitter.addWidget(self.plot_container)
        # Set stretch factors
        self.plot_splitter.setStretchFactor(0, 7)  # Plot gets 70%
        self.plot_splitter.setStretchFactor(1, 3)  # Annotations get 30%

        # Style the splitter handle
        self.plot_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #cccccc;
                border: 1px solid #999999;
            }
            QSplitter::handle:hover {
                background-color: #1f77b4;
            }
            QSplitter::handle:vertical {
                height: 5px;
            }
        """)
        
        # Setup annotation system (will add to splitter as SECOND item)
        self.setup_annotation_system(self.plot_splitter)
        
        # Set initial sizes (70% plot, 30% annotations)
        self.plot_splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(self.plot_splitter)
        
        return plot_widget
    
    def setup_annotation_system(self, parent_splitter):
        """Setup annotation system with drag-to-mark capability - FOR SPLITTER"""
        self.annotation_manager = AnnotationManager()
        self.current_time = 0.0
        
        # Enable drag-to-mark annotation
        self.drag_annotator = add_drag_to_mark_annotation(
            self.plot_widget, 
            self.annotation_manager
        )
        
        # Create annotation panel (will be added to splitter)
        annotation_panel = QWidget()
        annotation_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        annotation_layout = QVBoxLayout(annotation_panel)
        
        controls = AnnotationControls(self.annotation_manager, self)
        annotation_layout.addLayout(controls)
        
        self.annotation_widget = AnnotationWidget(self.annotation_manager)
        
        # Connect navigation signal
        self.annotation_widget.annotation_clicked.connect(self.navigate_to_annotation)
        
        annotation_layout.addWidget(self.annotation_widget)
        
        # Add annotation panel to splitter (SECOND item, below plot+slider)
        parent_splitter.addWidget(annotation_panel)

    def navigate_to_annotation(self, time):
        """Navigate to specific time in the plot"""
        if self.current_data is None:
            return
        
        # Find the annotation at this time
        annotation = None
        for ann in self.annotation_manager.annotations:
            if abs(ann.start_time - time) < 0.001:
                annotation = ann
                break
        
        if annotation and hasattr(self, 'plot_manager'):
            # Calculate appropriate time window based on annotation duration
            if annotation.is_point_event():
                window_duration = 1.0  # 1 second window for point events
            else:
                # Show 2x the duration of the event, or at least 0.5 seconds
                window_duration = max(0.5, annotation.duration * 2)
            
            # Center the view on the annotation
            center_time = (annotation.start_time + annotation.end_time) / 2
            start_time = max(0, center_time - window_duration / 2)
            end_time = start_time + window_duration
            
            # Update plot view
            self.plot_widget.setXRange(start_time, end_time, padding=0)
            
            print(f"Navigated to event at {annotation.start_time:.3f}s")

    def create_info_panel(self):
        """Create right info panel"""
        info_widget = QWidget()
        info_widget.setFixedWidth(250)
        info_layout = QVBoxLayout(info_widget)
        
        info_label = QLabel("Info")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        info_layout.addWidget(info_label)
        
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.setAlternatingRowColors(True)
        info_layout.addWidget(self.info_table)
        
        self.splitter.addWidget(info_widget)
    
    def create_menus(self):
        """Create simplified menu bar - File, Edit, View, Plot only"""
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # File menu
        file_menu = menubar.addMenu('&File')
        
        self.actionOpen = file_menu.addAction('&Open...')
        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.load_single_file)
        
        self.actionOpenMulti = file_menu.addAction('Load &Multiple Channels...')
        self.actionOpenMulti.setShortcut('Ctrl+Shift+O')
        self.actionOpenMulti.triggered.connect(self.load_multiple_channels)
        
        file_menu.addSeparator()
        
        self.actionClose = file_menu.addAction('&Close')
        self.actionClose.setShortcut('Ctrl+W')
        self.actionClose.triggered.connect(self.close_dataset)
        self.actionClose.setEnabled(False)
        
        file_menu.addSeparator()
        self.actionExit = file_menu.addAction('E&xit')
        self.actionExit.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')

        self.actionPreprocess = edit_menu.addAction('&Preprocess Data...')
        self.actionPreprocess.triggered.connect(self.preprocess_data)
        self.actionPreprocess.setEnabled(False)

        edit_menu.addSeparator()
        
        #sleep scoring
        self.actionLoadSleepScoring = edit_menu.addAction('Load &Sleep Scoring...')
        self.actionLoadSleepScoring.triggered.connect(self.load_sleep_scoring)
        self.actionLoadSleepScoring.setEnabled(False)
        
        edit_menu.addSeparator()
        
        self.actionChannelSelect = edit_menu.addAction('&Select Channels...')
        self.actionChannelSelect.triggered.connect(self.select_channels)
        self.actionChannelSelect.setEnabled(False)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        self.actionResetZoom = view_menu.addAction('&Reset Zoom')
        self.actionResetZoom.setShortcut('Ctrl+0')
        self.actionResetZoom.triggered.connect(self.reset_zoom)
        self.actionResetZoom.setEnabled(False)
        
        # Plot menu
        plot_menu = menubar.addMenu('&Plot')
    
        self.actionPlotData = plot_menu.addAction('Plot &Data')
        self.actionPlotData.triggered.connect(self.plot_data)
        self.actionPlotData.setEnabled(False)
        
        file_menu.addSeparator()

        self.actionNew = file_menu.addAction('&New Session')
        self.actionNew.setShortcut('Ctrl+N')
        self.actionNew.triggered.connect(self.reset_application_state)

    def apply_mnelab_styling(self):
        """Apply MNELAB-inspired styling"""
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
            QListWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                color: black;
                selection-background-color: #0078d4;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                color: black;
                gridline-color: #e0e0e0;
            }
            QLabel {
                color: black;
            }
        """)
    
    def setup_managers(self):
        """Setup manager classes"""
        self.multichannel_loader = MultiChannelLoader(self)
    
    def connect_signals(self):
        """Connect UI signals"""
        self.datasets_list.currentItemChanged.connect(self.dataset_changed)
        self.datasets_list.itemChanged.connect(self.dataset_renamed)
        self.datasets_list.itemClicked.connect(self.show_original_data)
        self.filtered_list.itemClicked.connect(self.show_filtered_data)

    def show_original_data(self, item):
        """Show original data when selected from datasets list"""
        index = item.data(Qt.UserRole)
        if index is not None:
            self.current_index = index
            self.current_data = self.datasets[index]["data"]
            self.current_header = self.datasets[index]["header"]
            
            self.update_info_panel()
            self.plot_data()
            self.filtered_list.clearSelection()
    
    def load_single_file(self):
        """Load single .continuous file with immediate downsample dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select .continuous file',
            filter="Continuous files (*.continuous)"
        )
        
        if not file_path:
            return
        
        try:
            # Load file
            result = load_single_channel(file_path, verbose=True)
            data = result['data']
            timestamps = result['timestamps']
            header = result['header']

            # Ensure timestamps start at zero
            if timestamps is not None and len(timestamps) > 0 and timestamps[0] != 0:
                print(f"Adjusting timestamps: original start = {timestamps[0]:.3f}s")
                timestamps = timestamps - timestamps[0]
                print(f"New timestamp range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
            print(f"Loaded file: {len(data)} samples at {header.get('sampleRate', 'Unknown')} Hz")
            
            # Show downsample dialog immediately
            self.show_downsample_dialog(data, header, timestamps, file_path)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")
            traceback.print_exc()
    
    def show_downsample_dialog(self, data, header, timestamps, file_path):
        """Show downsample dialog and handle downsampling"""
        dialog = DownsampleDialog(data, header, timestamps, file_path, self)
        
        if dialog.exec_() == QDialog.Accepted:
            target_fs = dialog.get_target_frequency()
            if target_fs is None:  # Invalid frequency
                return
                
            original_fs = header.get('sampleRate', 20000)
            
            print(f"Starting downsample from {original_fs:.0f}Hz to {target_fs:.0f}Hz")
            
            # Start downsampling
            self.start_downsampling(data, target_fs, original_fs, timestamps, file_path)
    
    def start_downsampling(self, data, target_fs, original_fs, timestamps, file_path):
        """Start downsampling with proper cleanup check"""
        try:
            # CRITICAL: Ensure clean state
            self.cleanup_worker_thread()
            
            # Verify we have clean state
            if self.worker_thread is not None:
                print("ERROR: Thread not cleaned up properly")
                return
            
            self.progress_dialog = QProgressDialog("Downsampling data...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()
            
            # Create NEW worker and thread objects
            self.worker = DownsampleWorker()
            self.worker.set_data(data, target_fs, original_fs, timestamps)
            
            self.worker_thread = QThread()  # Fresh new thread
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
                
                # Restore channel names from stored header
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
        
        print("\n" + "="*60)
        print("DEBUG: on_multichannel_loaded() - RECEIVED DATA")
        print("="*60)
        print(f"data.shape: {data.shape}")
        print(f"timestamps.shape: {timestamps.shape}")
        print(f"description: {description}")
        print(f"header keys: {list(header.keys()) if isinstance(header, dict) else 'Not a dict'}")
        if isinstance(header, dict) and 'channel_files' in header:
            print(f"channel_files value: {header['channel_files']}")
        print("="*60 + "\n")
        
        try:
            # Ensure timestamps start at zero
            if timestamps is not None and len(timestamps) > 0 and timestamps[0] != 0:
                timestamps = timestamps - timestamps[0]
            
            # Store for use in downsample completion
            self._multichannel_header = header
            
            # Show downsample dialog
            self.show_downsample_dialog(data, header, timestamps, description)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process multi-channel data:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def add_dataset(self, dataset):
        """Add dataset to the list"""
        self.datasets.append(dataset)
        print("\n" + "="*50)
        print("DATASET ADDED - DEBUG INFO")
        print("="*50)
        print(f"Name: {dataset['name']}")
        print(f"Data shape: {dataset['data'].shape}")
        print(f"Sample count: {len(dataset['data'])}")
        print(f"Timestamp count: {len(dataset.get('timestamps', []))}")
        print(f"Header sample rate: {dataset['header'].get('sampleRate', 'N/A')}")
        if 'channel_files' in dataset['header']:
            print(f"Source files: {dataset['header']['channel_files']}")
        print("="*50 + "\n")
        item = QListWidgetItem(dataset["name"])
        item.setData(Qt.UserRole, len(self.datasets) - 1)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.datasets_list.addItem(item)
        
        if self.sidebar_widget.isHidden():
            self.sidebar_widget.show()
        
        # Show info panel after data is loaded
        if self.splitter.widget(2).isHidden():
            self.splitter.widget(2).show()
        
        self.datasets_list.setCurrentItem(item)
        self.current_index = len(self.datasets) - 1
        self.current_data = dataset["data"]
        self.current_header = dataset["header"]
        
        self.content_stack.setCurrentWidget(self.plot_page)
        self.plot_data()
        self.update_ui_state()
    
    def dataset_changed(self, current, previous):
        """Handle dataset selection change"""
        if current:
            index = current.data(Qt.UserRole)
            self.current_index = index
            self.current_data = self.datasets[index]["data"]
            self.current_header = self.datasets[index]["header"]
            
            self.update_info_panel()
            self.plot_data()
    
    def dataset_renamed(self, item):
        """Handle dataset rename"""
        index = item.data(Qt.UserRole)
        self.datasets[index]["name"] = item.text()
    
    def show_filtered_data(self, item):
        """Show filtered data when selected"""
        dataset = item.data(Qt.UserRole)
        if dataset:
            self.current_data = dataset["data"]
            self.current_header = dataset["header"]
            
            timestamps = dataset.get("timestamps", 
                np.arange(len(self.current_data)) / self.current_header.get('sampleRate', 1000))
            
            if not hasattr(self, 'plot_manager'):
                self.plot_manager = PlotManager(self)
            
            self.plot_manager.plot_data(
                self.current_data, timestamps, self.current_header, dataset["file_path"]
            )
            
            self.update_info_panel()
            self.datasets_list.clearSelection()
    def preprocess_data(self):
        """Open preprocessing dialog - FIXED VERSION"""
        print("PREPROCESS_DATA METHOD CALLED!")
        import sys
        import traceback
        from datetime import datetime
        
        def log_error(msg):
            try:
                with open('app_errors.log', 'a') as f:
                    f.write(f"[{datetime.now()}] {msg}\n")
            except:
                pass
        
        log_error("Preprocess data method called")

        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        # FIXED: Safe thread checking
        try:
            if (hasattr(self, 'worker_thread') and 
                self.worker_thread is not None and 
                self.worker_thread.isRunning()):
                QMessageBox.warning(self, "Processing", "Already processing data. Please wait.")
                return
        except RuntimeError:
            # Thread object was deleted, safe to continue
            self.worker_thread = None
        
        # ==================================================================
        # IMPROVED CLEANUP: Only clear plot and remove slider, keep the page structure
        # ==================================================================
        if hasattr(self, 'comparison_plot_widget') and self.comparison_plot_widget is not None:
            # Clear the plot contents
            self.comparison_plot_widget.clear()
            print("✓ Cleared comparison plot")
        
        # Remove existing slider if present
        if hasattr(self, 'comparison_slider_widget') and self.comparison_slider_widget is not None:
            try:
                if hasattr(self, 'comparison_plot_slider_layout'):
                    self.comparison_plot_slider_layout.removeWidget(self.comparison_slider_widget)
                self.comparison_slider_widget.deleteLater()
                self.comparison_slider_widget = None
                print("✓ Removed old slider")
            except Exception as e:
                print(f"Warning: Could not remove slider: {e}")
        
        try:
            from dialogs import PreprocessingDialog
            
            if 'channel_files' not in self.current_header and self.current_index >= 0:
                # Extract from dataset file path
                file_path = self.datasets[self.current_index]["file_path"]
                if isinstance(file_path, str) and file_path.endswith('.continuous'):
                    channel_name = os.path.splitext(os.path.basename(file_path))[0]
                    self.current_header['channel_files'] = [f"{channel_name}.continuous"]
                    self.current_header['channel_count'] = 1
            
            dialog = PreprocessingDialog(
                self.current_data, 
                self.current_header, 
                self.datasets[self.current_index]["timestamps"] if self.current_index >= 0 else None, 
                self
            )
            
            if dialog.exec_() == QDialog.Accepted:
                params = dialog.get_parameters()
                if params and params.get('channels'):
                    self.start_robust_preprocessing(params)
                else:
                    QMessageBox.warning(self, "No Channels", "Please select at least one channel to process.")
                    
        except Exception as e:
            error_msg = f"Failed to open preprocessing dialog: {str(e)}\n{traceback.format_exc()}"
            log_error(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to open preprocessing dialog:\n{str(e)}")
            
    def start_robust_preprocessing(self, params):
        """Start preprocessing with comparison view output"""
        try:
            print("Starting preprocessing for comparison view...")
            
            # Clean up any existing thread FIRST
            self.cleanup_worker_thread()
            
            # NOW create progress dialog
            self.progress_dialog = QProgressDialog(
                "Preprocessing data...", 
                "Cancel", 
                0, 100, 
                self
            )
            self.progress_dialog.setWindowModality(Qt.NonModal)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.resize(400, 100)
            
            # Create worker object
            self.worker = PreprocessingWorker()
            self.worker.set_data(
                self.current_data,
                self.current_header,
                self.datasets[self.current_index]["timestamps"] if self.current_index >= 0 else None,
                params
            )
            
            # Create new thread
            self.worker_thread = QThread()
            self.worker.moveToThread(self.worker_thread)
            
            # Connect signals
            self.worker_thread.started.connect(self.worker.run)
            self.worker.progress_updated.connect(self.update_robust_progress)
            self.worker.processing_completed.connect(self.on_robust_preprocessing_complete)
            self.worker.error_occurred.connect(self.on_robust_preprocessing_error)
            self.progress_dialog.canceled.connect(self.cancel_robust_preprocessing)
            
            # Connect cleanup
            self.worker.processing_completed.connect(self.worker_thread.quit)
            self.worker.error_occurred.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            
            # Start processing
            self.worker_thread.start()
            # Progress dialog shows automatically due to setMinimumDuration(0)
            
            print("Preprocessing started successfully")
        
        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", 
                            f"Failed to start preprocessing:\n{str(e)}")
            traceback.print_exc()
        
    def on_robust_preprocessing_complete(self, raw_data, raw_timestamps, proc_data, 
                                    proc_timestamps, channel_names, original_fs, target_fs):
        """Handle successful preprocessing completion with comparison view"""
        try:
            print("=== PREPROCESSING COMPLETE - DEBUG ===")
            print(f"Raw data shape: {raw_data.shape}")
            print(f"Processed data shape: {proc_data.shape}")
            print(f"About to create comparison view...")
    
        
            # Close progress dialog
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            
            # Get real channel names
            real_channel_names = self.get_real_channel_names(len(channel_names))
            print(f"Real channel names: {real_channel_names}")
            
            # Calculate threshold using Karlsson & Frank method
            thresholds = []
            for i in range(proc_data.shape[1]):
                ch_data = proc_data[:, i]
                
                results = find_ripples_karlsson(
                    ch_data, 
                    fs=target_fs,
                    min_duration=0.015,
                    zscore_thresh=3.0,
                    smoothing_sigma=0.004,
                    f_plot=0
                )
                
                thresholds.append(results['thresh_envelope'])
                print(f"✓ Channel {i}: Threshold = {results['thresh_envelope']:.6f}")
                    
            # CREATE COMPARISON VIEW - this is the key step
            print("Calling create_comparison_view...")
            self.create_comparison_view(
                raw_data, raw_timestamps, proc_data, proc_timestamps,
                real_channel_names, original_fs, target_fs, thresholds=thresholds
            )
            print("✓ Comparison view created")
            
            # Store processed data in sidebar
            dataset = {
                "name": f"{self.datasets[self.current_index]['name']} (preprocessed-{len(real_channel_names)}ch)",
                "data": proc_data,
                "header": {
                    **self.current_header, 
                    'sampleRate': target_fs,
                    'channel_files': [f"{name}.continuous" for name in real_channel_names],  
                    'channel_count': len(real_channel_names) 
                },
                "file_path": f"Preprocessed {len(real_channel_names)} channels",
                "timestamps": proc_timestamps
            }
            item = QListWidgetItem(dataset['name'])
            item.setData(Qt.UserRole, dataset)
            self.filtered_list.addItem(item)
            print("✓ Added to filtered list")
            
            QMessageBox.information(self, "Preprocessing Complete", 
                                f"Processing successful!\n\n"
                                f"View shows before vs after preprocessing")
            
        except Exception as e:
            print(f"ERROR in preprocessing complete: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_real_channel_names(self, num_channels):
        """Extract real channel names from current dataset"""
        real_names = []
        
        try:
            # Method 1: Try to get from current dataset header
            if (self.current_index >= 0 and 
                'channel_files' in self.datasets[self.current_index]['header']):
                
                files = self.datasets[self.current_index]['header']['channel_files']
                print(f"✓ Found channel files in header: {files}")  # ADDED
                
                for i in range(min(num_channels, len(files))):
                    filename = files[i]
                    # Remove .continuous extension and clean up
                    clean_name = os.path.splitext(os.path.basename(filename))[0]
                    real_names.append(clean_name)
                    
            elif 'channel_files' in self.current_header:
                files = self.current_header['channel_files']
                print(f"✓ Found channel files in current header: {files}")  # ADDED
                
                for i in range(min(num_channels, len(files))):
                    filename = files[i]
                    clean_name = os.path.splitext(os.path.basename(filename))[0]
                    real_names.append(clean_name)
        
        except Exception as e:
            print(f"Error extracting channel names: {e}")
        
        # Fallback to generic names if nothing worked
        if len(real_names) < num_channels:
            print(f"⚠ Falling back to generic names for {num_channels - len(real_names)} channels")
            for i in range(len(real_names), num_channels):
                real_names.append(f"CH{i+1}")
        
        print(f"✓ Final channel names: {real_names}")  # ADDED
        return real_names[:num_channels]
    
    def cleanup_worker_thread(self):
        """Clean up existing worker thread"""
        if self.worker_thread and self.worker_thread.isRunning():
            if self.worker:
                self.worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.wait()
    
    def update_robust_progress(self, percentage, message):
        """Update progress dialog"""
        if self.progress_dialog and not self.progress_dialog.wasCanceled():
            self.progress_dialog.setValue(percentage)
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()
    
    def on_robust_preprocessing_error(self, error_message):
        """Handle preprocessing error"""
        print(f"Preprocessing error: {error_message}")
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        QMessageBox.critical(self, "Preprocessing Error", 
                           f"Processing failed:\n{error_message}")
    
    def cancel_robust_preprocessing(self):
        """Cancel preprocessing"""
        print("Cancelling preprocessing...")
        
        if self.worker:
            self.worker.cancel()
        
        self.cleanup_worker_thread()
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
    
    def create_comparison_view(self, raw_data, raw_timestamps, proc_data, proc_timestamps, 
                      channel_names, original_fs, target_fs, thresholds=None):
        """Create comparison view with amplitude control support - FULLY FIXED VERSION"""
        
        try:
            QApplication.processEvents()
            
            if not hasattr(self, 'comparison_page'):
                self.comparison_page = QWidget()
                main_layout = QVBoxLayout(self.comparison_page)
                main_layout.setContentsMargins(0, 0, 0, 0)
                
                # ==================================================================
                # CREATE THE PLOT WIDGET FIRST (before referencing it!)
                # ==================================================================
                self.comparison_plot_widget = pg.PlotWidget()
                self.comparison_plot_widget.setBackground('w')
                self.comparison_plot_widget.showGrid(x=True, y=True, alpha=0.3)
                self.comparison_plot_widget.setLabel('bottom', 'Time (seconds)', units='')
                self.comparison_plot_widget.setTitle("Before/After Preprocessing Comparison")
                
                # Performance optimizations
                self.comparison_plot_widget.setClipToView(True)
                self.comparison_plot_widget.setDownsampling(auto=True, mode='peak')
                self.comparison_plot_widget.setMouseEnabled(x=True, y=True)
                self.comparison_plot_widget.getPlotItem().setClipToView(True)
                
                # ==================================================================
                # CREATE RESIZABLE SPLITTER with correct configuration
                # ==================================================================
                self.comparison_splitter = QSplitter(Qt.Vertical)
                self.comparison_splitter.setHandleWidth(5)  # Make handle visible
                self.comparison_splitter.setChildrenCollapsible(False)  # Prevent collapsing
                
                # ==================================================================
                # TOP SECTION: Container for plot + slider
                # ==================================================================
                plot_and_slider_container = QWidget()
                plot_and_slider_container.setSizePolicy(
                    QSizePolicy.Expanding, 
                    QSizePolicy.Expanding  # Allow resizing
                )
                plot_slider_layout = QVBoxLayout(plot_and_slider_container)
                plot_slider_layout.setContentsMargins(0, 0, 0, 0)
                plot_slider_layout.setSpacing(0)
                
                # Add plot to container
                plot_slider_layout.addWidget(self.comparison_plot_widget)
                
                # Store layout reference for slider addition later
                self.comparison_plot_slider_layout = plot_slider_layout
                
                # ==================================================================
                # BOTTOM SECTION: Annotation panel
                # ==================================================================
                annotation_panel = QWidget()
                annotation_panel.setSizePolicy(
                    QSizePolicy.Expanding,
                    QSizePolicy.Expanding  # Allow resizing
                )
                annotation_layout = QVBoxLayout(annotation_panel)
                
                controls = AnnotationControls(self.annotation_manager, self)
                annotation_layout.addLayout(controls)
                
                self.comparison_annotation_widget = AnnotationWidget(self.annotation_manager)
                annotation_layout.addWidget(self.comparison_annotation_widget)
                
                # ==================================================================
                # ADD BOTH SECTIONS TO SPLITTER
                # ==================================================================
                self.comparison_splitter.addWidget(plot_and_slider_container)  # Top
                self.comparison_splitter.addWidget(annotation_panel)           # Bottom
                
                # Set initial sizes (70% plot area, 30% annotations)
                self.comparison_splitter.setSizes([700, 300])
                
                # Set stretch factors (both can resize)
                self.comparison_splitter.setStretchFactor(0, 7)  # Top section gets 70%
                self.comparison_splitter.setStretchFactor(1, 3)  # Bottom section gets 30%
                
                # Style the splitter handle to make it visible
                self.comparison_splitter.setStyleSheet("""
                    QSplitter::handle {
                        background-color: #cccccc;
                        border: 1px solid #999999;
                    }
                    QSplitter::handle:hover {
                        background-color: #1f77b4;
                    }
                    QSplitter::handle:vertical {
                        height: 5px;
                    }
                """)
                
                # Add splitter to main layout
                main_layout.addWidget(self.comparison_splitter)
                
                # Add to content stack
                self.content_stack.addWidget(self.comparison_page)
            
            # Switch to comparison view
            self.content_stack.setCurrentWidget(self.comparison_page)
            
            # Setup drag annotator if not exists
            if not hasattr(self, 'comparison_drag_annotator'):
                self.comparison_drag_annotator = add_drag_to_mark_annotation(
                    self.comparison_plot_widget, self.annotation_manager
                )
            
            # Clear existing plots
            self.comparison_plot_widget.clear()
            
            # Store data for controls and amplitude scaling
            self.comparison_raw_data = raw_data
            self.comparison_raw_timestamps = raw_timestamps
            self.comparison_proc_data = proc_data
            self.comparison_proc_timestamps = proc_timestamps
            self.comparison_channel_names = channel_names
            
            n_channels = min(len(channel_names), raw_data.shape[1], proc_data.shape[1])
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # Calculate spacing
            all_data = np.concatenate([raw_data[:, :n_channels], proc_data[:, :n_channels]], axis=1)
            max_range = np.max(np.ptp(all_data, axis=0))
            self.comparison_spacing = max_range * 2
            
            # Store base normalized data for amplitude scaling
            self.comparison_raw_normalized = []
            self.comparison_proc_normalized = []
            self.comparison_raw_curves = []
            self.comparison_proc_curves = []
            
            # Plot BEFORE (top) - store curves
            for i in range(n_channels):
                y_offset = (n_channels + 1) * self.comparison_spacing + (n_channels - 1 - i) * self.comparison_spacing
                
                ch_data = raw_data[:, i]
                normalized_base = (ch_data / np.ptp(ch_data)) * self.comparison_spacing * 0.8
                self.comparison_raw_normalized.append(normalized_base)
                
                normalized = normalized_base + y_offset
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                
                curve = self.comparison_plot_widget.plot(
                    raw_timestamps, normalized, 
                    pen=pen,
                    antialias=False,
                    clipToView=True,
                    autoDownsample=True,
                    downsampleMethod='peak'
                )
                self.comparison_raw_curves.append(curve)
            
            # Separator
            separator_y = n_channels * self.comparison_spacing + self.comparison_spacing/2
            pen_sep = pg.mkPen(color='gray', width=3, style=Qt.SolidLine)
            self.comparison_separator = self.comparison_plot_widget.plot(
                [raw_timestamps[0], raw_timestamps[-1]], 
                [separator_y, separator_y], 
                pen=pen_sep
            )
            
            # Plot AFTER (bottom) - store curves
            for i in range(n_channels):
                y_offset = (n_channels - 1 - i) * self.comparison_spacing
                
                ch_data = proc_data[:, i]
                normalized_base = (ch_data / np.ptp(ch_data)) * self.comparison_spacing * 0.8
                self.comparison_proc_normalized.append(normalized_base)
                
                normalized = normalized_base + y_offset
                pen = pg.mkPen(color=colors[i % len(colors)], width=2, style=Qt.SolidLine)
                
                curve = self.comparison_plot_widget.plot(
                    proc_timestamps, normalized,
                    pen=pen,
                    antialias=False,
                    clipToView=True,
                    autoDownsample=True,
                    downsampleMethod='peak'
                )
                self.comparison_proc_curves.append(curve)

            if thresholds and len(thresholds) == n_channels:
                self.comparison_threshold_lines = []
                self.comparison_threshold_labels = []
                
                for i in range(n_channels):
                    y_offset = (n_channels - 1 - i) * self.comparison_spacing
                    ch_data = proc_data[:, i]
                    
                    # Get envelope threshold value
                    threshold_value = thresholds[i]
                    
                    # Calculate the envelope of THIS processed data to get scale
                    from scipy.signal import hilbert
                    instantaneous_amplitude = np.abs(hilbert(ch_data))
                    envelope_ptp = np.ptp(instantaneous_amplitude)
                    
                    # Normalize threshold using ENVELOPE scale, not signal scale
                    threshold_normalized = (threshold_value / envelope_ptp) * self.comparison_spacing * 0.8
                    threshold_y = y_offset + threshold_normalized
                    
                    # Draw threshold line
                    pen_thresh = pg.mkPen(color='red', width=2, style=Qt.DashLine)
                    threshold_line = self.comparison_plot_widget.plot(
                        [proc_timestamps[0], proc_timestamps[-1]],
                        [threshold_y, threshold_y],
                        pen=pen_thresh
                    )
                    self.comparison_threshold_lines.append(threshold_line)
                    
                    # Add label
                    text_label = pg.TextItem(
                        text=f"Threshold (3σ): {threshold_value:.2f}",
                        color=(255, 0, 0),
                        anchor=(0, 0.5),
                        fill=(255, 255, 255, 200)
                    )
                    text_label.setPos(proc_timestamps[0] + (proc_timestamps[-1] - proc_timestamps[0]) * 0.02, threshold_y)
                    self.comparison_plot_widget.addItem(text_label)
                    self.comparison_threshold_labels.append(text_label)

        # Add channel name labels on Y-axis
            # Add channel name labels on Y-axis
            y_axis = self.comparison_plot_widget.getAxis('left')
            ticks = []

            # BEFORE section (top)
            for i in range(n_channels):
                y_offset = (n_channels + 1) * self.comparison_spacing + (n_channels - 1 - i) * self.comparison_spacing
                ticks.append((y_offset, f"{channel_names[i]} (Before)"))

            # Separator
            separator_y = n_channels * self.comparison_spacing + self.comparison_spacing/2
            ticks.append((separator_y, ""))

            # AFTER section (bottom)
            for i in range(n_channels):
                y_offset = (n_channels - 1 - i) * self.comparison_spacing
                ticks.append((y_offset, f"{channel_names[i]} (After)"))

            y_axis.setTicks([ticks])
            y_axis.setLabel('')
            y_axis.setTextPen(pg.mkPen(color='#999999')) 
            y_axis.setPen(pg.mkPen(color='#CCCCCC', width=1))  
            
            x_axis = self.comparison_plot_widget.getAxis('bottom')
            x_axis.enableAutoSIPrefix(False)
            x_axis.setLabel('Time (seconds)', units='')
            
            # Set ranges
            n_channels = len(self.comparison_raw_normalized)
            total_height = (n_channels * 2 + 1) * self.comparison_spacing
            self.comparison_plot_widget.setYRange(-self.comparison_spacing, total_height, padding=0.1)
            
            time_start = min(raw_timestamps[0], proc_timestamps[0])
            time_end = max(raw_timestamps[-1], proc_timestamps[-1])
            self.comparison_plot_widget.setXRange(time_start, time_end, padding=0.02)
            
            # Setup controls
            self._setup_comparison_controls()
            if self.sleep_scoring_data is not None:
                print("Adding sleep scoring to comparison view...")
                self.add_sleep_scoring_to_comparison()
    
            QApplication.processEvents()
            print("✓✓✓ Comparison view created with RESIZABLE splitter!")
            
        except Exception as e:
            import traceback
            from datetime import datetime
            
            error_msg = f"Error creating comparison view: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Log to file
            try:
                with open('app_errors.log', 'a') as f:
                    f.write(f"[{datetime.now()}] {error_msg}\n")
            except:
                pass
            
            QMessageBox.critical(self, "Error", 
                f"Failed to create comparison view:\n{str(e)}\n\n"
                "Check app_errors.log for details.")

    def close_dataset(self):
        """Close current dataset and allow loading new data"""
        if self.current_index >= 0:
            reply = QMessageBox.question(
                self, "Close Dataset",
                "Close current dataset and clear all data?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Clean up threading first
                self.cleanup_worker_thread()
                
                # Clear current dataset
                current_item = self.datasets_list.currentItem()
                if current_item:
                    row = self.datasets_list.row(current_item)
                    self.datasets_list.takeItem(row)
                    self.datasets.pop(self.current_index)
                    
                    # Update indices
                    for i in range(self.datasets_list.count()):
                        item = self.datasets_list.item(i)
                        if item.data(Qt.UserRole) > self.current_index:
                            item.setData(Qt.UserRole, item.data(Qt.UserRole) - 1)
                
                # Reset state
                self.current_index = -1
                self.current_data = None
                self.current_header = None
                
                # Clear filtered data
                self.filtered_list.clear()
                self.filtered_datasets = []
                
                # Reset UI
                if len(self.datasets) == 0:
                    self.content_stack.setCurrentWidget(self.empty_page)
                    self.sidebar_widget.hide()
                    self.splitter.widget(2).hide()  # Hide info panel
                    self.clear_info_panel()
                    
                    # Clear plots
                    if hasattr(self, 'plot_widget'):
                        self.plot_widget.clear()
                    if hasattr(self, 'comparison_plot_widget'):
                        self.comparison_plot_widget.clear()
                else:
                    if self.datasets_list.count() > 0:
                        self.datasets_list.setCurrentRow(0)
                
                self.update_ui_state()
                print("✓ Dataset closed - ready for new data")

    def close_dataset(self):
        """Close current dataset and allow loading new data"""
        if self.current_index >= 0:
            reply = QMessageBox.question(
                self, "Close Dataset",
                "Close current dataset and clear all data?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Clean up threading first
                self.cleanup_worker_thread()
                
                # Clear current dataset
                current_item = self.datasets_list.currentItem()
                if current_item:
                    row = self.datasets_list.row(current_item)
                    self.datasets_list.takeItem(row)
                    self.datasets.pop(self.current_index)
                    
                    # Update indices
                    for i in range(self.datasets_list.count()):
                        item = self.datasets_list.item(i)
                        if item.data(Qt.UserRole) > self.current_index:
                            item.setData(Qt.UserRole, item.data(Qt.UserRole) - 1)
                
                # Reset state
                self.current_index = -1
                self.current_data = None
                self.current_header = None
                
                # Clear filtered data
                self.filtered_list.clear()
                self.filtered_datasets = []
                
                # Reset UI
                if len(self.datasets) == 0:
                    self.content_stack.setCurrentWidget(self.empty_page)
                    self.sidebar_widget.hide()
                    self.splitter.widget(2).hide()  # Hide info panel
                    self.clear_info_panel()
                    
                    # Clear plots
                    if hasattr(self, 'plot_widget'):
                        self.plot_widget.clear()
                    if hasattr(self, 'comparison_plot_widget'):
                        self.comparison_plot_widget.clear()
                else:
                    if self.datasets_list.count() > 0:
                        self.datasets_list.setCurrentRow(0)
                
                self.update_ui_state()
                print("✓ Dataset closed - ready for new data")

    def reset_application_state(self):
        """Complete application reset"""
        print("Resetting application state...")
        
        # CRITICAL: Thread cleanup
        self.cleanup_worker_thread()
        
        # Clear all data
        self.datasets.clear()
        self.filtered_datasets.clear()
        self.datasets_list.clear()
        self.filtered_list.clear()
        
        self.current_data = None
        self.current_header = None
        self.current_index = -1
        
        # Reset UI
        self.content_stack.setCurrentWidget(self.empty_page)
        self.sidebar_widget.hide()
        self.splitter.widget(2).hide()
        self.clear_info_panel()
        
        if hasattr(self, 'plot_widget'):
            self.plot_widget.clear()
        if hasattr(self, 'comparison_plot_widget'):
            self.comparison_plot_widget.clear()
        
        # Remove toolbars
        if hasattr(self, 'zoom_toolbar'):
            self.removeToolBar(self.zoom_toolbar)
        if hasattr(self, 'comparison_toolbar'):
            self.removeToolBar(self.comparison_toolbar)
        
        self.update_ui_state()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("✓ Application reset complete")
    
    def plot_data(self):
        """Plot current dataset"""
        if self.current_data is not None and self.current_header is not None:
            if not hasattr(self, 'plot_manager'):
                self.plot_manager = PlotManager(self)
            
            if self.current_index >= 0:
                timestamps = self.datasets[self.current_index]["timestamps"]
                file_path = self.datasets[self.current_index]["file_path"]
            else:
                timestamps = np.arange(len(self.current_data)) / self.current_header.get('sampleRate', 1000)
                file_path = "Processed data"
            
            self.plot_manager.plot_data(
                self.current_data, timestamps, self.current_header, file_path
            )
    
    def update_info_panel(self):
        """Update info panel with current dataset info - emphasize sampling rate"""
        if self.current_data is None:
            self.clear_info_panel()
            return
        
        data = self.current_data
        header = self.current_header
        
        if self.current_index >= 0:
            file_name = os.path.basename(self.datasets[self.current_index]["file_path"])
            timestamps = self.datasets[self.current_index]["timestamps"]
        else:
            file_name = "Processed data"
            timestamps = None
        
        # Calculate duration from timestamps if available
        if timestamps is not None and len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            start_time = 0.0 
        else:
            duration = len(data) / header.get('sampleRate', 1000)
            start_time = 0.0
        
        # Get sampling rate with emphasis
        sampling_rate = header.get('sampleRate', 'Unknown')
        if isinstance(sampling_rate, (int, float)):
            rate_display = f"{sampling_rate:.0f} Hz"
            rate_status = "Downsampled" if sampling_rate <= 5000 else "Downsampled"
        else:
            rate_display = str(sampling_rate)
            rate_status = "Unknown"
        
        info_dict = {
            "File name": file_name,
            "Sampling Rate": f"{rate_display} ({rate_status})",  # Emphasized
            "Data type": "Processed" if header.get('sampleRate', 1000) <= 5000 else "Raw",
            "Channels": f"{data.shape[1] if data.ndim > 1 else 1}",
            "Samples": f"{len(data):,}",
            "Duration": f"{duration:.2f} s",
            "Start time": f"{start_time:.3f} s",
            "Size in memory": f"{data.nbytes / 1024 / 1024:.2f} MB",
        }
        
        self.info_table.setRowCount(len(info_dict))
        for i, (key, value) in enumerate(info_dict.items()):
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            
            # Highlight sampling rate row
            if "Sampling Rate" in key:
                font = QFont()
                font.setBold(True)
                key_item.setFont(font)
                value_item.setFont(font)
                
                # Color based on rate
                if isinstance(sampling_rate, (int, float)):
                    if sampling_rate <= 2000:
                        key_item.setBackground(pg.mkColor('#d1ecf1'))  # Light blue
                        value_item.setBackground(pg.mkColor('#d1ecf1'))
                    elif sampling_rate <= 5000:
                        key_item.setBackground(pg.mkColor('#d4edda'))  # Light green
                        value_item.setBackground(pg.mkColor('#d4edda'))
                    else:
                        key_item.setBackground(pg.mkColor('#fff3cd'))  # Light yellow
                        value_item.setBackground(pg.mkColor('#fff3cd'))
            
            self.info_table.setItem(i, 0, key_item)
            self.info_table.setItem(i, 1, value_item)
        
        self.info_table.resizeColumnsToContents()
    
    def clear_info_panel(self):
        """Clear info panel"""
        self.info_table.setRowCount(0)
    
    def update_ui_state(self):
        """Update UI state based on loaded data"""
        has_data = self.current_data is not None
        
        self.actionClose.setEnabled(has_data)
        self.actionPreprocess.setEnabled(has_data) 
        self.actionChannelSelect.setEnabled(has_data)
        self.actionPlotData.setEnabled(has_data)
        self.actionResetZoom.setEnabled(has_data)
        self.actionLoadSleepScoring.setEnabled(has_data)
    
        print(f"DEBUG: Preprocess menu enabled = {self.actionPreprocess.isEnabled()}")
    def load_sleep_scoring(self):
        """Load sleep scoring data from .mat file"""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", 
                              "Please load neural data first before loading sleep scoring.")
            return
        
        # Get duration and sampling rate from current data
        if self.current_index >= 0:
            timestamps = self.datasets[self.current_index]["timestamps"]
            if timestamps is not None:
                duration = timestamps[-1] - timestamps[0]
            else:
                duration = len(self.current_data) / self.current_header.get('sampleRate', 1000)
        else:
            duration = len(self.current_data) / self.current_header.get('sampleRate', 1000)
        
        fs = self.current_header.get('sampleRate', 1000)
        
        # Open dialog
        dialog = SleepScoringDialog(
            neural_data_duration=duration,
            neural_fs=fs,
            parent=self 
    )
        
        if dialog.exec_() == QDialog.Accepted:
            self.sleep_scoring_data = dialog.get_sleep_scoring_data()
            
            if self.sleep_scoring_data:
                unique_states = len(np.unique(self.sleep_scoring_data['states']))
                QMessageBox.information(
                    self, "Sleep Scoring Loaded",
                    f"Sleep scoring loaded successfully!\n\n"
                f"Duration: {duration:.1f} seconds\n"
                    f"States: {unique_states} unique states\n\n"
                f"Sleep scoring will appear in comparison view when you preprocess data."           
                )
            
            
                print(f"Sleep scoring loaded: {len(self.sleep_scoring_data['states'])} samples")

    def add_sleep_scoring_to_comparison(self):
        """Add sleep scoring track to comparison view"""
        if self.sleep_scoring_data is None or not hasattr(self, 'comparison_plot_widget'):
            return
        
        try:
            states = self.sleep_scoring_data['states']
            state_names = self.sleep_scoring_data['state_names']
            
            # Create timestamps for sleep scoring
            if hasattr(self, 'comparison_raw_timestamps') and self.comparison_raw_timestamps is not None:
                # Align with neural data timestamps
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.linspace(
                    self.comparison_raw_timestamps[0],
                    self.comparison_raw_timestamps[-1],
                    len(states)
                )
            else:
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.arange(len(states)) / fs
            
            # Define colors for each state (RGBA format)
            state_colors = {
               0: (220, 220, 220, 100),  # Unscored - Light Grey
               
                1: (200, 230, 200, 120),  # Awake - Soft Green
                3: (200, 220, 240, 120),  # Non-REM - Soft Blue
                5: (240, 210, 220, 120),  # REM - Soft Pink
                4: (230, 210, 240, 120)   # Intermediate - Soft Purple
            }
            # Calculate y-position for sleep scoring track
            n_channels = len(self.comparison_channel_names) if hasattr(self, 'comparison_channel_names') else 1
            spacing = self.comparison_spacing if hasattr(self, 'comparison_spacing') else 100
            
            # Sleep scoring track at the very top
            track_height = spacing * 0.8
            track_y_position = (n_channels * 2 + 3) * spacing
            
            # Remove old sleep scoring items if they exist
            if hasattr(self, 'sleep_scoring_items') and self.sleep_scoring_items:
                for item in self.sleep_scoring_items:
                    try:
                        self.comparison_plot_widget.removeItem(item)
                    except:
                        pass
            
            self.sleep_scoring_items = []
            
            # Plot sleep states as colored rectangles
            current_state = states[0]
            start_idx = 0
            
            for i in range(1, len(states) + 1):
                # Check if state changed or end of data
                if i == len(states) or states[i] != current_state:
                    # Draw rectangle for this state period
                    start_time = timestamps[start_idx]
                    end_time = timestamps[i-1] if i < len(timestamps) else timestamps[-1]
                    
                    # Get color for this state
                    color = state_colors.get(current_state, (128, 128, 128, 100))
                    
                    # Create filled rectangle using LinearRegionItem
                    from pyqtgraph import LinearRegionItem
                    region = LinearRegionItem(
                        values=[start_time, end_time],
                        orientation='vertical',
                        brush=pg.mkBrush(color),
                        movable=False,
                        pen=pg.mkPen(None)
                    )
                    
                    # Set the region to the correct y-range
                    region.setZValue(-10)  # Behind the data
                    self.comparison_plot_widget.addItem(region)
                    self.sleep_scoring_items.append(region)
                    
                    # Update for next segment
                    if i < len(states):
                        current_state = states[i]
                        start_idx = i
            
            # Add label at the top
            label_item = pg.TextItem(
                "Sleep States:",
                anchor=(0, 0.5),
                color=(0, 0, 0)
            )
            label_item.setFont(QFont("Arial", 10, QFont.Bold))
            label_item.setPos(timestamps[0], track_y_position + track_height * 0.6)
            self.comparison_plot_widget.addItem(label_item)
            self.sleep_scoring_items.append(label_item)
            
            # Add legend for states
            legend_x = timestamps[0] + (timestamps[-1] - timestamps[0]) * 0.02
            legend_y = track_y_position + track_height * 0.3
            
            for state_val, color in state_colors.items():
                if state_val in np.unique(states):
                    # Create small colored box
                    box_width = (timestamps[-1] - timestamps[0]) * 0.015
                    box = pg.QtGui.QGraphicsRectItem(
                        legend_x, 
                        legend_y - track_height * 0.15,
                        box_width,
                        track_height * 0.3
                    )
                    box.setBrush(pg.mkBrush(color))
                    box.setPen(pg.mkPen((0, 0, 0), width=1))
                    self.comparison_plot_widget.addItem(box)
                    self.sleep_scoring_items.append(box)
                    
                    # Add text label
                    text = pg.TextItem(
                        state_names.get(state_val, f'State {state_val}'),
                        anchor=(0, 0.5),
                        color=(0, 0, 0)
                    )
                    text.setPos(legend_x + box_width * 1.5, legend_y)
                    self.comparison_plot_widget.addItem(text)
                    self.sleep_scoring_items.append(text)
                    
                    # Move to next legend position
                    legend_x += (timestamps[-1] - timestamps[0]) * 0.12
            
            # Update y-axis range to include sleep scoring
            current_y_range = self.comparison_plot_widget.viewRange()[1]
            new_max = max(current_y_range[1], track_y_position + track_height * 1.2)
            self.comparison_plot_widget.setYRange(
                current_y_range[0], 
                new_max, 
                padding=0.05
            )
            
            print("Sleep scoring track added to comparison view")
            print(f"   States displayed: {np.unique(states)}")
            print(f"   Duration: {timestamps[-1] - timestamps[0]:.1f} seconds")
            
        except Exception as e:
            print(f"Error adding sleep scoring: {e}")
            import traceback
            traceback.print_exc()
    def select_channels(self):
        """Open channel selection dialog"""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        n_channels = self.current_data.shape[1] if self.current_data.ndim > 1 else 1
        channel_names = [f"Channel {i+1}" for i in range(n_channels)]
        
        dialog = ChannelSelectionDialog(channel_names, list(range(min(8, n_channels))), self)
        if dialog.exec_() == QDialog.Accepted:
            selected_channels = dialog.get_selected_channels()
            
            if self.current_data.ndim == 1:
                selected_data = self.current_data
            else:
                selected_data = self.current_data[:, selected_channels]
            
            selected_dataset = {
                "name": f"{self.datasets[self.current_index]['name']} (selected-{len(selected_channels)}ch)",
                "data": selected_data,
                "header": self.current_header,
                "file_path": "Selected channels",
                "timestamps": self.datasets[self.current_index]["timestamps"]
            }
            
            self.add_dataset(selected_dataset)
        
    def on_processing_error(self, error_msg):
        """Handle processing errors"""
        QMessageBox.critical(self, "Processing Error", f"Error occurred:\n{error_msg}")
    
    def reset_zoom(self):
        """Reset plot zoom"""
        if hasattr(self, 'plot_manager'):
            self.plot_manager.reset_zoom()
        self.setUpdatesEnabled(True)

    def closeEvent(self, event):
        """Clean up on application close"""
        self.cleanup_worker_thread()
        super().closeEvent(event)
        
    def _setup_comparison_controls(self):
        """Setup zoom controls for comparison view WITH amp controls"""
        print(f" _setup_comparison_controls called!")

        # FORCEFULLY hide PlotManager's toolbar
        if hasattr(self, 'plot_manager') and hasattr(self.plot_manager, 'parent'):
            # Find and hide the zoom toolbar created by PlotManager
            for toolbar in self.findChildren(QToolBar):
                # PlotManager creates toolbar with "Zoom" in title
                if "Zoom" in toolbar.windowTitle() or "Main" in toolbar.windowTitle():
                    print(f"  → Hiding PlotManager toolbar: {toolbar.windowTitle()}")
                    toolbar.setVisible(False)
        
        # Also hide by direct reference if it exists
        if hasattr(self, 'zoom_toolbar') and self.zoom_toolbar:
            print(f"  → Hiding zoom_toolbar directly")
            self.zoom_toolbar.setVisible(False)
        
        # Remove any old comparison toolbars
        if hasattr(self, 'comparison_toolbar') and self.comparison_toolbar is not None:
            self.removeToolBar(self.comparison_toolbar)
            self.comparison_toolbar.deleteLater()
            self.comparison_toolbar = None
        
        # Create toolbar
        toolbar = QToolBar("Comparison Controls")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self.comparison_toolbar = toolbar
        
        toolbar.setStyleSheet("""
            QToolBar { 
                spacing: 8px; 
                padding: 5px;
                background: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
            QPushButton { 
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
                min-width: 70px;
            }
            QPushButton:hover {
                background: #e8f4fd;
                border: 1px solid #1f77b4;
            }
            QComboBox {
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
                min-width: 80px;
            }
        """)
        
        # Zoom buttons
        btn = QPushButton("🔍+ Zoom In")
        btn.clicked.connect(self._comparison_zoom_in)
        toolbar.addWidget(btn)
        
        btn = QPushButton("🔍- Zoom Out")
        btn.clicked.connect(self._comparison_zoom_out)
        toolbar.addWidget(btn)
        
        btn = QPushButton("Reset")
        btn.clicked.connect(self._comparison_reset)
        toolbar.addWidget(btn)
        
        toolbar.addSeparator()
        
        # AMPLITUDE CONTROLS
        btn_amp_in = QPushButton("+ Amp")
        btn_amp_in.clicked.connect(self._comparison_amp_increase)
        toolbar.addWidget(btn_amp_in)
        
        btn_amp_out = QPushButton("- Amp")
        btn_amp_out.clicked.connect(self._comparison_amp_decrease)
        toolbar.addWidget(btn_amp_out)
        
        # Amplitude scale label
        self.comparison_amp_label = QLabel("Amp: 1.0x")
        self.comparison_amp_label.setStyleSheet("""
            QLabel {
                color: #333;
                padding: 4px 8px;
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 3px;
            }
        """)
        toolbar.addWidget(self.comparison_amp_label)
        
        toolbar.addSeparator()
        
        # Time window dropdown
        toolbar.addWidget(QLabel("Time Window:"))
        self.comparison_time_combo = QComboBox()
        self.comparison_time_combo.addItems([
            "Full View", "10 min", "5 min", "1 min", "30 sec", "10 sec", 
            "1 sec", "100 ms", "10 ms"
        ])
        self.comparison_time_combo.currentTextChanged.connect(self._comparison_time_window_changed)
        toolbar.addWidget(self.comparison_time_combo)
        
        toolbar.addSeparator()
        
        # View info label
        self.comparison_view_label = QLabel("View: Full")
        self.comparison_view_label.setStyleSheet("""
            QLabel {
                color: #333;
                padding: 4px 8px;
                background: #e8f4fd;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        toolbar.addWidget(self.comparison_view_label)
        
        toolbar.addSeparator()
        
        # Back button
        btn = QPushButton("← Back to Data View")
        btn.clicked.connect(self._back_to_data_view)
        toolbar.addWidget(btn)
        
        # Add slider below the plot
        self._add_comparison_slider()
        
        # Initialize flags and amplitude scale
        self.comparison_slider_updating = False
        self.comparison_updating = False
        self.comparison_amp_scale = 1.0
        
        # Connect view change signal
        try:
            # Disconnect first to avoid duplicates
            self.comparison_plot_widget.sigRangeChanged.disconnect()
        except:
            pass
        
        self.comparison_plot_widget.sigRangeChanged.connect(self._comparison_view_changed)
        
        print("✓ _setup_comparison_controls COMPLETE")

    def _add_comparison_slider(self):
        """Add navigation slider to comparison view - FIXED to place above annotations"""
        
        # ==================================================================
        # FIX: Use the stored layout reference instead of searching
        # ==================================================================
        
        if not hasattr(self, 'comparison_plot_slider_layout'):
            print("Warning: comparison_plot_slider_layout not found!")
            return
        
        # Remove any existing slider first
        if hasattr(self, 'comparison_slider_widget') and self.comparison_slider_widget is not None:
            try:
                self.comparison_plot_slider_layout.removeWidget(self.comparison_slider_widget)
                self.comparison_slider_widget.deleteLater()
            except:
                pass
        
        # Create slider widget
        from PyQt5.QtWidgets import QSlider
        self.comparison_slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.comparison_slider_widget)
        slider_layout.setContentsMargins(5, 0, 5, 5)
        
        self.comparison_slider = QSlider(Qt.Horizontal)
        self.comparison_slider.setMinimum(0)
        self.comparison_slider.setMaximum(1000)
        self.comparison_slider.setValue(0)
        self.comparison_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1f77b4;
                border: 2px solid #0d5aa7;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #2e8bc0;
            }
            QSlider::sub-page:horizontal {
                background: #d0e8ff;
                border-radius: 4px;
            }
        """)
        self.comparison_slider.setMaximumHeight(30)
        self.comparison_slider.valueChanged.connect(self._comparison_slider_moved)
        
        slider_layout.addWidget(self.comparison_slider)
        
        # Add slider to the plot_slider_layout (below plot, above annotations)
        self.comparison_plot_slider_layout.addWidget(self.comparison_slider_widget)
        
        print("✓ Slider added to correct location (below plot, above annotations)")

    def _comparison_slider_moved(self, val):
        """Handle slider movement"""
        if self.comparison_slider_updating or not hasattr(self, 'comparison_raw_timestamps'):
            return
        
        xrange = self.comparison_plot_widget.viewRange()[0]
        width = xrange[1] - xrange[0]
        
        time_start = self.comparison_raw_timestamps[0]
        time_end = self.comparison_raw_timestamps[-1]
        total_duration = time_end - time_start
        
        relative_pos = val / 1000.0
        center = time_start + relative_pos * total_duration
        
        new_start = max(time_start, center - width/2)
        new_end = min(time_end, center + width/2)
        
        if new_start == time_start:
            new_end = min(time_end, time_start + width)
        elif new_end == time_end:
            new_start = max(time_start, time_end - width)
        
        self.comparison_plot_widget.sigRangeChanged.disconnect()
        self.comparison_plot_widget.setXRange(new_start, new_end, padding=0)
        self.comparison_plot_widget.sigRangeChanged.connect(self._comparison_view_changed)
        
        self._update_comparison_view_label()

    def _comparison_view_changed(self):
        """Update slider when view changes"""
        if self.comparison_slider_updating or not hasattr(self, 'comparison_raw_timestamps'):
            return
        
        xrange = self.comparison_plot_widget.viewRange()[0]
        center = (xrange[0] + xrange[1]) / 2
        
        time_start = self.comparison_raw_timestamps[0]
        time_end = self.comparison_raw_timestamps[-1]
        total_duration = time_end - time_start
        
        relative_pos = (center - time_start) / total_duration
        relative_pos = max(0.0, min(1.0, relative_pos))
        slider_val = int(relative_pos * 1000)
        
        self.comparison_slider_updating = True
        self.comparison_slider.setValue(slider_val)
        self.comparison_slider_updating = False
        
        self._update_comparison_view_label()

    def _comparison_time_window_changed(self, text):
        """Apply selected time window"""
        if not hasattr(self, 'comparison_raw_timestamps'):
            return
        
        time_start = self.comparison_raw_timestamps[0]
        time_end = self.comparison_raw_timestamps[-1]
        
        if text == "Full View":
            self.comparison_plot_widget.setXRange(time_start, time_end, padding=0.02)
        else:
            xrange = self.comparison_plot_widget.viewRange()[0]
            current_center = (xrange[0] + xrange[1]) / 2
            
            # Parse duration
            if "min" in text:
                duration = float(text.split()[0]) * 60
            elif "sec" in text:
                duration = float(text.split()[0])
            elif "ms" in text:
                duration = float(text.split()[0]) / 1000.0
            else:
                duration = time_end - time_start
            
            new_start = max(time_start, current_center - duration/2)
            new_end = min(time_end, current_center + duration/2)
            
            if new_start == time_start:
                new_end = min(time_end, time_start + duration)
            elif new_end == time_end:
                new_start = max(time_start, time_end - duration)
            
            self.comparison_plot_widget.setXRange(new_start, new_end, padding=0)
        
        self._update_comparison_view_label()

    def _update_comparison_view_label(self):
        """Update view info label"""
        if not hasattr(self, 'comparison_view_label'):
            return
        
        xrange = self.comparison_plot_widget.viewRange()[0]
        dur = xrange[1] - xrange[0]
        
        if dur >= 60:
            txt = f"View: {dur/60:.1f} min"
        elif dur >= 1:
            txt = f"View: {dur:.1f} sec"
        else:
            txt = f"View: {dur*1000:.1f} ms"
        
        self.comparison_view_label.setText(txt)

    def _comparison_zoom_in(self):
        print("DEBUG: _comparison_zoom_in called!")
        
        # Temporarily disconnect signal
        self.comparison_plot_widget.sigRangeChanged.disconnect()
        
        try:
            xrange = self.comparison_plot_widget.viewRange()[0]
            center = (xrange[0] + xrange[1]) / 2
            width = (xrange[1] - xrange[0]) / 2
            self.comparison_plot_widget.setXRange(center - width/2, center + width/2, padding=0)
            self._update_comparison_view_label()
        finally:
            # Reconnect signal
            self.comparison_plot_widget.sigRangeChanged.connect(self._comparison_view_changed)

    def _comparison_zoom_out(self):
        """Zoom out on comparison view"""
        xrange = self.comparison_plot_widget.viewRange()[0]
        center = (xrange[0] + xrange[1]) / 2
        width = (xrange[1] - xrange[0]) * 2
        
        if hasattr(self, 'comparison_raw_timestamps'):
            max_width = self.comparison_raw_timestamps[-1] - self.comparison_raw_timestamps[0]
            width = min(width, max_width)
        
        self.comparison_plot_widget.setXRange(center - width/2, center + width/2, padding=0)

    def _comparison_reset(self):
        """Reset comparison view including amplitude"""
        if self.comparison_updating:
            return
            
        self.comparison_updating = True
        
        try:
            # Reset zoom
            if hasattr(self, 'comparison_raw_timestamps'):
                time_start = self.comparison_raw_timestamps[0]
                time_end = self.comparison_raw_timestamps[-1]
                
                self.comparison_plot_widget.sigRangeChanged.disconnect()
                self.comparison_plot_widget.setXRange(time_start, time_end, padding=0.02)
                self.comparison_plot_widget.getViewBox().updateAutoRange()
                self.comparison_plot_widget.sigRangeChanged.connect(self._comparison_view_changed)
            
            # Reset amplitude
            self.comparison_amp_scale = 1.0
            self._update_comparison_amplitudes()
            self.comparison_amp_label.setText("Amp: 1.0x")
            
            # Reset dropdown
            if hasattr(self, 'comparison_time_combo'):
                self.comparison_time_combo.setCurrentText("Full View")
            
            self._update_comparison_view_label()
            
        finally:
            self.comparison_updating = False

    def _comparison_amp_increase(self):
        """Increase amplitude scaling in comparison view"""
        self.comparison_amp_scale *= 1.2
        self._update_comparison_amplitudes()
        self.comparison_amp_label.setText(f"Amp: {self.comparison_amp_scale:.1f}x")

    def _comparison_amp_decrease(self):
        """Decrease amplitude scaling in comparison view"""
        self.comparison_amp_scale /= 1.2
        self._update_comparison_amplitudes()
        self.comparison_amp_label.setText(f"Amp: {self.comparison_amp_scale:.1f}x")

    def _update_comparison_amplitudes(self):
        """Update all traces with new amplitude scaling"""
        if not hasattr(self, 'comparison_raw_curves'):
            return
        
        n_channels = len(self.comparison_raw_normalized)
        
        # Update BEFORE traces (top section)
        for i, (curve, base_data) in enumerate(zip(self.comparison_raw_curves, self.comparison_raw_normalized)):
            y_offset = (n_channels + 1) * self.comparison_spacing + (n_channels - 1 - i) * self.comparison_spacing
            scaled_data = base_data * self.comparison_amp_scale + y_offset
            curve.setData(self.comparison_raw_timestamps, scaled_data)
        
        # Update separator
        separator_y = n_channels * self.comparison_spacing + self.comparison_spacing/2
        if hasattr(self, 'comparison_separator'):
            self.comparison_separator.setData(
                [self.comparison_raw_timestamps[0], self.comparison_raw_timestamps[-1]],
                [separator_y, separator_y]
            )
        
        # Update AFTER traces (bottom section)
        for i, (curve, base_data) in enumerate(zip(self.comparison_proc_curves, self.comparison_proc_normalized)):
            y_offset = (n_channels - 1 - i) * self.comparison_spacing
            scaled_data = base_data * self.comparison_amp_scale + y_offset
            curve.setData(self.comparison_proc_timestamps, scaled_data)
            
    def _back_to_data_view(self):
        """Return to main data view"""
        self.content_stack.setCurrentWidget(self.plot_page)
        
        if hasattr(self, 'comparison_toolbar'):
            self.removeToolBar(self.comparison_toolbar)
        
        if hasattr(self, 'plot_manager'):
            self.plot_data()

    def reset_application_state(self):
        """Reset application state with proper thread cleanup"""
        print("Resetting application state...")
        
        # CRITICAL: Proper thread cleanup
        self.cleanup_worker_thread()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear current data
        self.current_data = None
        self.current_header = None
        self.current_index = -1
        
        # Reset UI
        self.content_stack.setCurrentWidget(self.empty_page)
        self.clear_info_panel()
        
        if hasattr(self, 'plot_widget'):
            self.plot_widget.clear()
        
        self.update_ui_state()
        print("Reset complete")

    def cleanup_worker_thread(self):
        """ROBUST thread cleanup to prevent Qt deletion errors"""
        print("Cleaning up worker threads...")
        
        # Close progress dialog first
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            try:
                self.progress_dialog.close()
                self.progress_dialog.deleteLater()
            except:
                pass
            self.progress_dialog = None
        
        # Clean up worker
        if hasattr(self, 'worker') and self.worker:
            try:
                if hasattr(self.worker, 'cancel'):
                    self.worker.cancel()
                self.worker.deleteLater()
            except:
                pass
            self.worker = None
        
        # Clean up thread - CRITICAL PART
        if hasattr(self, 'worker_thread') and self.worker_thread:
            try:
                # Disconnect all signals first
                self.worker_thread.started.disconnect()
                self.worker_thread.finished.disconnect()
                
                if self.worker_thread.isRunning():
                    self.worker_thread.quit()
                    if not self.worker_thread.wait(2000):  # Wait max 2 seconds
                        self.worker_thread.terminate()
                        self.worker_thread.wait()
                
                self.worker_thread.deleteLater()
            except RuntimeError:
                # Thread already deleted - this is fine
                pass
            except:
                # Any other error - still safe to continue
                pass
            
            self.worker_thread = None
        
        print("Thread cleanup complete")

    def test_preprocess_menu(self):
        """Test method to verify menu connection works"""
        print("SUCCESS: Menu item clicked and connected properly!")
        QMessageBox.information(self, "Menu Test", 
                            "Edit → Preprocess menu is working!\n\n"
                            "The menu connection is fine. Check if:\n"
                            "1. Data is loaded\n"
                            "2. Menu item is enabled\n"
                            "3. dialogs.py is in the correct location")
def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
        
    app.setApplicationName("Neural Data Analysis")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Neural Data Analysis")
    
    window = OpenEphysMainWindow()
    window.show()
    
    print("OpenEphysLab started with complete downsample + preprocess workflow!")
    print("Workflow: File → Open → Choose downsample rate → Edit → Preprocess → Comparison view")
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()