import sys
import os
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QSplitter, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QStackedWidget,
    QMessageBox, QDialog, QProgressDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont

import pyqtgraph as pg
import numpy as np

# Local modules
from PlotManager import PlotManager
from MultiChannelLoader import MultiChannelLoader
from annotation import (
    AnnotationManager, AnnotationWidget, AnnotationControls,
    add_drag_to_mark_annotation
)
from dialogs import ChannelSelectionDialog, PreprocessingDialog
from workers import PreprocessingWorker
from comparison_view import comparison_view
from sleep_scoring_mixin import SleepScoringMixin
from data_loader import DataLoader

class OpenEphysMainWindow(QMainWindow, comparison_view, SleepScoringMixin, DataLoader):
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
        self.sleep_legend = self.create_sleep_scoring_legend_widget()
        info_layout.addWidget(self.sleep_legend)
        
    
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
    def run_ripple_detection(self):
        """Placeholder for ripple detection - to be implemented"""
        QMessageBox.information(self, "Ripple Detection", 
                               "Ripple detection will be implemented here.\n\n"
                               "This will use the Karlsson & Frank method.")
    
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
            
    def add_dataset(self, dataset):
        """Add dataset to the list"""
        self.datasets.append(dataset)
        print("\n" + "="*50)
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
            print("Cleared comparison plot")
        
        # Remove existing slider if present
        if hasattr(self, 'comparison_slider_widget') and self.comparison_slider_widget is not None:
            try:
                if hasattr(self, 'comparison_plot_slider_layout'):
                    self.comparison_plot_slider_layout.removeWidget(self.comparison_slider_widget)
                self.comparison_slider_widget.deleteLater()
                self.comparison_slider_widget = None
                print("Removed old slider")
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
        
    def on_robust_preprocessing_complete(self, raw_data, raw_timestamps, proc_data, proc_timestamps, channel_names, original_fs, target_fs):
        #"Handle successful preprocessing completion with comparison view
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
            
            # ================================================================
            # Calculate thresholds for RAW data (BEFORE section)
            # ================================================================
            from scipy.signal import hilbert
            
            raw_thresholds = []
            for i in range(raw_data.shape[1]):
                ch_data = raw_data[:, i]
                envelope = np.abs(hilbert(ch_data))
                threshold_value = np.mean(envelope) + 3.0 * np.std(envelope)
                raw_thresholds.append(threshold_value)
                print(f"Raw Channel {i}: Threshold = {threshold_value:.6f}")
            
            # ================================================================
            # Calculate thresholds for PROCESSED data (AFTER section)
            # ================================================================
            proc_thresholds = []
            for i in range(proc_data.shape[1]):
                ch_data = proc_data[:, i]
                envelope = np.abs(hilbert(ch_data))
                threshold_value = np.mean(envelope) + 3.0 * np.std(envelope)
                proc_thresholds.append(threshold_value)
                print(f"Proc Channel {i}: Threshold = {threshold_value:.6f}")

            # CREATE COMPARISON VIEW - pass both threshold lists
            print("Calling create_comparison_view...")
            self.create_comparison_view(
                raw_data, raw_timestamps, proc_data, proc_timestamps,
                real_channel_names, original_fs, target_fs,
                raw_thresholds=raw_thresholds, proc_thresholds=proc_thresholds
            )
            print("Comparison view created")
            
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
            print("Added to filtered list")
            
            QMessageBox.information(self, "Preprocessing Complete", 
                                f"Processing successful!\n\n"
                                f"View shows before vs after preprocessing\n"
                                f"Red dashed lines = ripple detection threshold (3σ)")
            
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
                print(f"Found channel files in header: {files}")  # ADDED
                
                for i in range(min(num_channels, len(files))):
                    filename = files[i]
                    # Remove .continuous extension and clean up
                    clean_name = os.path.splitext(os.path.basename(filename))[0]
                    real_names.append(clean_name)
                    
            elif 'channel_files' in self.current_header:
                files = self.current_header['channel_files']
                print(f" Found channel files in current header: {files}")  # ADDED
                
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
        
        print(f"Final channel names: {real_names}")  # ADDED
        return real_names[:num_channels]
    
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
                print(" Dataset closed - ready for new data")

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
        print("Application reset complete")
    
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
        if hasattr(self, 'actionRippleDetect'):
            self.actionRippleDetect.setEnabled(has_data)
        self.actionLoadSleepScoring.setEnabled(has_data)

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
        
    def cleanup_worker_thread(self):
        """ROBUST thread cleanup to prevent Qt deletion errors"""

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