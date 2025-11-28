"""
Integration Code for Adding Sleep Scoring to Comparison View

This file shows the modifications needed to integrate sleep scoring
into your existing comparison view in main.py
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QPushButton, QMessageBox, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# ============================================================================
# STEP 1: Add this import at the top of main.py
# ============================================================================
"""
from sleep_scoring_dialog import SleepScoringDialog
"""

# ============================================================================
# STEP 2: Add these instance variables to __init__ in OpenEphysMainWindow
# ============================================================================
"""
def __init__(self):
    super().__init__()
    
    # ... existing code ...
    
    # ADD THESE LINES:
    self.sleep_scoring_data = None  # Store sleep scoring information
    self.sleep_scoring_plot_item = None  # Plot item for sleep scoring
"""

# ============================================================================
# STEP 3: Add menu item for loading sleep scoring (in create_menus)
# ============================================================================
"""
def create_menus(self):
    # ... existing menu code ...
    
    # In the Edit menu, after actionPreprocess:
    edit_menu.addSeparator()
    
    self.actionLoadSleepScoring = edit_menu.addAction('Load &Sleep Scoring...')
    self.actionLoadSleepScoring.triggered.connect(self.load_sleep_scoring)
    self.actionLoadSleepScoring.setEnabled(False)  # Enable when data is loaded
"""

# ============================================================================
# STEP 4: Add method to load sleep scoring
# ============================================================================
"""
def load_sleep_scoring(self):
    '''Load sleep scoring data from .mat file'''
    if self.current_data is None:
        QMessageBox.warning(self, "No Data", 
                          "Please load neural data first before loading sleep scoring.")
        return
    
    # Get duration and sampling rate from current data
    if self.current_index >= 0:
        timestamps = self.datasets[self.current_index]["timestamps"]
        duration = timestamps[-1] - timestamps[0] if timestamps is not None else len(self.current_data) / self.current_header.get('sampleRate', 1000)
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
            QMessageBox.information(
                self, "Sleep Scoring Loaded",
                f"Sleep scoring loaded successfully!\\n\\n"
                f"Duration: {duration:.1f} seconds\\n"
                f"States: {len(np.unique(self.sleep_scoring_data['states']))} unique states\\n\\n"
                f"Sleep scoring will appear in comparison view."
            )
            
            # If comparison view exists, update it
            if hasattr(self, 'comparison_plot_widget'):
                self.add_sleep_scoring_to_comparison()
"""

# ============================================================================
# STEP 5: Modify create_comparison_view to include sleep scoring section
# ============================================================================
"""
def create_comparison_view(self, raw_data, raw_timestamps, proc_data, proc_timestamps, 
                          channel_names, original_fs, target_fs):
    '''Create comparison view with sleep scoring support'''
    
    # ... existing setup code ...
    
    # Create plot widget (MODIFY THIS SECTION)
    self.comparison_plot_widget = pg.PlotWidget()
    self.comparison_plot_widget.setBackground('w')
    self.comparison_plot_widget.showGrid(x=True, y=True, alpha=0.3)
    
    # IMPORTANT: Enable linking across multiple plot items
    self.comparison_plot_widget.setLabel('bottom', 'Time (seconds)', units='')
    
    layout.addWidget(self.comparison_plot_widget)
    
    # ... rest of plotting code ...
    
    # ADD THIS AT THE END, BEFORE switching to comparison view:
    # Add sleep scoring if available
    if self.sleep_scoring_data is not None:
        self.add_sleep_scoring_to_comparison()
    
    # ... rest of method ...
"""

# ============================================================================
# STEP 6: Add method to plot sleep scoring
# ============================================================================
def add_sleep_scoring_to_comparison(self):
    """Add sleep scoring track to comparison view"""
    if self.sleep_scoring_data is None or not hasattr(self, 'comparison_plot_widget'):
        return
    
    try:
        states = self.sleep_scoring_data['states']
        state_names = self.sleep_scoring_data['state_names']
        
        # Create timestamps for sleep scoring
        if hasattr(self, 'comparison_raw_timestamps'):
            timestamps = self.comparison_raw_timestamps[:len(states)]
            if len(timestamps) < len(states):
                # Extend timestamps if needed
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.arange(len(states)) / fs
        else:
            fs = self.sleep_scoring_data.get('sampling_rate', 1000)
            timestamps = np.arange(len(states)) / fs
        
        # Define colors for each state - Pastel & Faded
        state_colors = {
            0: (220, 220, 220, 100),  # Unscored - Light Grey
            1: (200, 230, 200, 120),  # Awake - Soft Green
            3: (200, 220, 240, 120),  # Non-REM - Soft Blue
            5: (240, 210, 220, 120),  # REM - Soft Pink
            4: (230, 210, 240, 120)   # Intermediate - Soft Purple
        }
        
        # Calculate y-position for sleep scoring track
        # Place it at the top, above the "Before" section
        n_channels = len(self.comparison_channel_names) if hasattr(self, 'comparison_channel_names') else 1
        spacing = self.comparison_spacing if hasattr(self, 'comparison_spacing') else 100
        
        # Sleep scoring track height
        track_height = spacing * 0.5
        track_y_position = (n_channels * 2 + 2) * spacing  # Above everything
        
        # Remove old sleep scoring if it exists
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
                
                # Create rectangle
                rect = pg.QtGui.QGraphicsRectItem(
                    start_time,
                    track_y_position - track_height/2,
                    end_time - start_time,
                    track_height
                )
                rect.setBrush(pg.mkBrush(color))
                rect.setPen(pg.mkPen(None))
                
                self.comparison_plot_widget.addItem(rect)
                self.sleep_scoring_items.append(rect)
                
                # Update for next segment
                if i < len(states):
                    current_state = states[i]
                    start_idx = i
        
        # Add label and legend
        self.add_sleep_scoring_label(track_y_position, track_height, state_names)
        
        # Update y-axis range to include sleep scoring
        self.comparison_plot_widget.setYRange(
            -spacing, 
            track_y_position + track_height, 
            padding=0.1
        )
        
        print("âœ… Sleep scoring added to comparison view")
        
    except Exception as e:
        print(f"Error adding sleep scoring: {e}")
        import traceback
        traceback.print_exc()

def add_sleep_scoring_label(self, y_position, height, state_names):
    """Add label and legend for sleep scoring track"""
    
    # Add text label
    text_item = pg.TextItem(
        "Sleep States",
        anchor=(0, 0.5),
        color=(0, 0, 0)
    )
    text_item.setPos(-50, y_position)  # Adjust position as needed
    self.comparison_plot_widget.addItem(text_item)
    
    if hasattr(self, 'sleep_scoring_items'):
        self.sleep_scoring_items.append(text_item)
    
    # Add to Y-axis ticks
    if hasattr(self, 'comparison_plot_widget'):
        y_axis = self.comparison_plot_widget.getAxis('left')
        
        # Get existing ticks
        existing_ticks = y_axis.tickValues(0, 1000, 10)  # Dummy values to get structure
        
        # Try to add sleep scoring tick
        try:
            # This is a simplified version - you may need to adjust based on your exact setup
            pass
        except:
            pass

# ============================================================================
# STEP 7: Update UI state to enable sleep scoring menu
# ============================================================================
"""
def update_ui_state(self):
    '''Update UI state based on loaded data'''
    has_data = self.current_data is not None
    
    self.actionClose.setEnabled(has_data)
    self.actionPreprocess.setEnabled(has_data) 
    self.actionChannelSelect.setEnabled(has_data)
    self.actionPlotData.setEnabled(has_data)
    self.actionResetZoom.setEnabled(has_data)
    
    # ADD THIS LINE:
    self.actionLoadSleepScoring.setEnabled(has_data)
"""

# ============================================================================
# STEP 8: Example of complete modified create_comparison_view
# ============================================================================

def create_comparison_view_with_sleep_scoring(
    self, raw_data, raw_timestamps, proc_data, proc_timestamps, 
    channel_names, original_fs, target_fs
):
    """Complete example of comparison view with sleep scoring"""
    
    # Create page if doesn't exist
    if not hasattr(self, 'comparison_page'):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
        
        self.comparison_page = QWidget()
        layout = QVBoxLayout(self.comparison_page)
        
        # Create plot widget
        self.comparison_plot_widget = pg.PlotWidget()
        self.comparison_plot_widget.setBackground('w')
        self.comparison_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.comparison_plot_widget.setLabel('bottom', 'Time (seconds)', units='')
        self.comparison_plot_widget.setTitle("Before/After Preprocessing with Sleep Scoring")
        
        # Performance optimizations
        self.comparison_plot_widget.setClipToView(True)
        self.comparison_plot_widget.setDownsampling(auto=True, mode='peak')
        
        layout.addWidget(self.comparison_plot_widget)
        
        # Add annotation panel (from your existing code)
        from PyQt5.QtWidgets import QWidget as QW
        annotation_panel = QW()
        from PyQt5.QtWidgets import QVBoxLayout as QVL
        annotation_layout = QVL(annotation_panel)
        
        from annotation import AnnotationControls, AnnotationWidget
        controls = AnnotationControls(self.annotation_manager, self)
        annotation_layout.addLayout(controls)
        
        self.comparison_annotation_widget = AnnotationWidget(self.annotation_manager)
        annotation_layout.addWidget(self.comparison_annotation_widget)
        
        layout.addWidget(annotation_panel)
        
        self.content_stack.addWidget(self.comparison_page)
    
    # Switch to comparison view
    self.content_stack.setCurrentWidget(self.comparison_page)
    
    # Setup drag annotator
    if not hasattr(self, 'comparison_drag_annotator'):
        from annotation import add_drag_to_mark_annotation
        self.comparison_drag_annotator = add_drag_to_mark_annotation(
            self.comparison_plot_widget, self.annotation_manager
        )
    
    self.comparison_plot_widget.clear()
    
    # Store data
    self.comparison_raw_data = raw_data
    self.comparison_raw_timestamps = raw_timestamps
    self.comparison_proc_data = proc_data
    self.comparison_proc_timestamps = proc_timestamps
    self.comparison_channel_names = channel_names
    
    n_channels = min(len(channel_names), raw_data.shape[1], proc_data.shape[1])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Calculate spacing (accounting for sleep scoring track)
    all_data = np.concatenate([raw_data[:, :n_channels], proc_data[:, :n_channels]], axis=1)
    max_range = np.max(np.ptp(all_data, axis=0))
    self.comparison_spacing = max_range * 2
    
    # Reserve space for sleep scoring track at top
    sleep_track_space = self.comparison_spacing * 1.5
    
    # Store base normalized data
    self.comparison_raw_normalized = []
    self.comparison_proc_normalized = []
    self.comparison_raw_curves = []
    self.comparison_proc_curves = []
    
    # Plot BEFORE (middle section)
    for i in range(n_channels):
        y_offset = (n_channels + 1) * self.comparison_spacing + (n_channels - 1 - i) * self.comparison_spacing
        y_offset += sleep_track_space  # Shift down for sleep scoring
        
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
    separator_y = n_channels * self.comparison_spacing + self.comparison_spacing/2 + sleep_track_space
    pen_sep = pg.mkPen(color='gray', width=3, style=Qt.SolidLine)
    self.comparison_separator = self.comparison_plot_widget.plot(
        [raw_timestamps[0], raw_timestamps[-1]], 
        [separator_y, separator_y], 
        pen=pen_sep
    )
    
    # Plot AFTER (bottom section)
    for i in range(n_channels):
        y_offset = (n_channels - 1 - i) * self.comparison_spacing + sleep_track_space
        
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
    
    # Add channel labels
    y_axis = self.comparison_plot_widget.getAxis('left')
    ticks = []
    
    # Sleep scoring label (will be added by add_sleep_scoring_to_comparison)
    
    # BEFORE section
    for i in range(n_channels):
        y_offset = (n_channels + 1) * self.comparison_spacing + (n_channels - 1 - i) * self.comparison_spacing + sleep_track_space
        ticks.append((y_offset, f"{channel_names[i]} (Before)"))
    
    # Separator
    ticks.append((separator_y, ""))
    
    # AFTER section
    for i in range(n_channels):
        y_offset = (n_channels - 1 - i) * self.comparison_spacing + sleep_track_space
        ticks.append((y_offset, f"{channel_names[i]} (After)"))
    
    y_axis.setTicks([ticks])
    y_axis.setLabel('')
    
    # Set ranges
    total_height = (n_channels * 2 + 1) * self.comparison_spacing + sleep_track_space * 2
    self.comparison_plot_widget.setYRange(-self.comparison_spacing, total_height, padding=0.1)
    
    time_start = min(raw_timestamps[0], proc_timestamps[0])
    time_end = max(raw_timestamps[-1], proc_timestamps[-1])
    self.comparison_plot_widget.setXRange(time_start, time_end, padding=0.02)
    
    # Setup controls
    self._setup_comparison_controls()
    
    # ADD SLEEP SCORING if available
    if self.sleep_scoring_data is not None:
        self.add_sleep_scoring_to_comparison()
    
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
    
    print("âœ… Comparison view with sleep scoring created")


# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================
"""
To integrate sleep scoring into your application:

1. Copy sleep_scoring_dialog.py to your project directory

2. In main.py, add these imports:
   from sleep_scoring_dialog import SleepScoringDialog

3. Add instance variables in __init__:
   self.sleep_scoring_data = None
   self.sleep_scoring_items = []

4. Add menu item in create_menus():
   self.actionLoadSleepScoring = edit_menu.addAction('Load &Sleep Scoring...')
   self.actionLoadSleepScoring.triggered.connect(self.load_sleep_scoring)

5. Add the load_sleep_scoring() method (see above)

6. Add the add_sleep_scoring_to_comparison() method (see above)

7. Add the add_sleep_scoring_label() method (see above)

8. Modify create_comparison_view() to call add_sleep_scoring_to_comparison()

9. Update update_ui_state() to enable the menu item

10. Test by:
    - Loading neural data
    - Loading sleep scoring from .mat file
    - Opening comparison view to see both neural data and sleep states
"""