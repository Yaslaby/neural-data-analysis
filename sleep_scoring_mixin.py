"""
Sleep Scoring Mixin Module
Provides sleep scoring visualization and loading functionality for OpenEphysMainWindow.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QFrame,
    QMessageBox, QDialog
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

STATE_COLORS = {
    0: (220, 220, 220, 100),  # Unscored - Light Grey
    1: (200, 230, 200, 120),  # Awake - Soft Green
    3: (200, 220, 240, 120),  # Non-REM - Soft Blue
    4: (230, 210, 240, 120),  # Intermediate - Soft Purple
    5: (240, 210, 220, 120),  # REM - Soft Pink 
    }

STATE_NAMES = {
    0: 'Unscored',
    1: 'Awake',
    3: 'Non-REM',
    4: 'Intermediate',
    5: 'REM'
    
}

class SleepScoringMixin:
    
    def init_sleep_scoring(self):
        """Initialize sleep scoring attributes. Call this in __init__."""
        self.sleep_scoring_data = None
        self.sleep_scoring_items = []
    
    # -------------------------------------------------------------------------
    # Loading Methods
    # -------------------------------------------------------------------------
    
    def load_sleep_scoring(self):
        """Load sleep scoring data from .mat file via dialog."""
        if self.current_data is None:
            QMessageBox.warning(
                self, "No Data",
                "Please load neural data first before loading sleep scoring."
            )
            return
        
        # Calculate duration from current data
        if self.current_index >= 0:
            timestamps = self.datasets[self.current_index]["timestamps"]
            if timestamps is not None:
                duration = timestamps[-1] - timestamps[0]
            else:
                duration = len(self.current_data) / self.current_header.get('sampleRate', 1000)
        else:
            duration = len(self.current_data) / self.current_header.get('sampleRate', 1000)
        
        fs = self.current_header.get('sampleRate', 1000)
        
        # Import dialog here to avoid circular imports
        from sleep_scoring_dialog import SleepScoringDialog
        
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
                    f"Sleep scoring will appear in comparison view."
                )
                
                # Update comparison view if it exists
                if hasattr(self, 'comparison_plot_widget'):
                    self.add_sleep_scoring_to_comparison()
    
    # -------------------------------------------------------------------------
    # Visualization Methods
    # -------------------------------------------------------------------------
    
    def add_sleep_scoring_to_comparison(self):
        """
        Add sleep scoring visualization to comparison view.
        Call this at the end of create_comparison_view().
        """
        if self.sleep_scoring_data is None:
            return
        
        if not hasattr(self, 'comparison_plot_widget'):
            return
        
        try:
            states = self.sleep_scoring_data['states']
            state_names = self.sleep_scoring_data.get('state_names', STATE_NAMES)
            state_colors = self.sleep_scoring_data.get('state_colors', STATE_COLORS)
            
            # Create timestamps aligned with neural data
            if hasattr(self, 'comparison_raw_timestamps') and self.comparison_raw_timestamps is not None:
                timestamps = np.linspace(
                    self.comparison_raw_timestamps[0],
                    self.comparison_raw_timestamps[-1],
                    len(states)
                )
            else:
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.arange(len(states)) / fs
            
            # Remove old sleep scoring items
            self._clear_sleep_scoring_items()
            
            # Draw sleep states as vertical regions (full plot height)
            self._draw_sleep_state_regions(states, timestamps, state_colors)
            
            # Add legend at top of plot
            self._add_sleep_scoring_legend(timestamps, states, state_names, state_colors)
            
            print(" Sleep scoring added to comparison view")
            
        except Exception as e:
            print(f"Error adding sleep scoring: {e}")
            import traceback
            traceback.print_exc()
    
    def _clear_sleep_scoring_items(self):
        """Remove existing sleep scoring plot items."""
        if hasattr(self, 'sleep_scoring_items') and self.sleep_scoring_items:
            for item in self.sleep_scoring_items:
                try:
                    self.comparison_plot_widget.removeItem(item)
                except:
                    pass
        self.sleep_scoring_items = []
    
    def _draw_sleep_state_regions(self, states, timestamps, state_colors):
        """Draw colored vertical regions for each sleep state period."""
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states) + 1):
            # Check if state changed or reached end
            if i == len(states) or states[i] != current_state:
                start_time = timestamps[start_idx]
                end_time = timestamps[i - 1] if i < len(timestamps) else timestamps[-1]
                
                color = state_colors.get(current_state, (128, 128, 128, 80))
                
                # Use LinearRegionItem for full-height vertical bands
                region = pg.LinearRegionItem(
                    values=[start_time, end_time],
                    orientation='vertical',
                    brush=pg.mkBrush(color),
                    movable=False,
                    pen=pg.mkPen(None)
                )
                region.setZValue(-10)  # Behind neural traces
                
                self.comparison_plot_widget.addItem(region)
                self.sleep_scoring_items.append(region)
                
                # Update for next segment
                if i < len(states):
                    current_state = states[i]
                    start_idx = i
    
    def _add_sleep_scoring_legend(self, timestamps, states, state_names, state_colors):
        """Add legend showing sleep state colors at top of plot."""
        # Calculate position for legend (top of plot)
        n_channels = len(self.comparison_channel_names) if hasattr(self, 'comparison_channel_names') else 1
        spacing = self.comparison_spacing if hasattr(self, 'comparison_spacing') else 100
        
        legend_y = (n_channels * 2 + 2) * spacing
        legend_x = timestamps[0] + (timestamps[-1] - timestamps[0]) * 0.01
        
    # -------------------------------------------------------------------------
    # Widget Creation Methods
    # -------------------------------------------------------------------------
    
    def create_sleep_scoring_legend_widget(self):
        """
        Create a standalone legend widget for sleep states.
        Can be added to sidebars or dialogs.
        
        Returns:
            QWidget: Legend widget showing all sleep state colors
        """
        legend_widget = QWidget()
        legend_widget.setMaximumHeight(180)
        layout = QVBoxLayout(legend_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Title
        title = QLabel("Sleep State Colors")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 3px; background: #f0f8ff; border: 1px solid #ccc;")
        layout.addWidget(title)
        
        # Color grid
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(2, 2, 2, 2)
        
        for row, (state_code, color) in enumerate(sorted(STATE_COLORS.items())):
            name = STATE_NAMES.get(state_code, f'State {state_code}')
            
            # Color square
            color_frame = QFrame()
            color_frame.setFixedSize(22, 22)
            color_frame.setFrameStyle(QFrame.Box | QFrame.Plain)
            color_frame.setLineWidth(1)
            
            palette = color_frame.palette()
            palette.setColor(QPalette.Window, QColor(color[0], color[1], color[2], color[3]))
            color_frame.setPalette(palette)
            color_frame.setAutoFillBackground(True)
            
            grid_layout.addWidget(color_frame, row, 0)
            
            # State name
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 10))
            grid_layout.addWidget(name_label, row, 1)
        
        layout.addWidget(grid_widget)
        layout.addStretch()
        
        return legend_widget
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def has_sleep_scoring(self):
        """Check if sleep scoring data is loaded."""
        return self.sleep_scoring_data is not None
    
    def clear_sleep_scoring(self):
        """Clear all sleep scoring data and visualization."""
        self._clear_sleep_scoring_items()
        self.sleep_scoring_data = None
    
    def get_sleep_state_at_time(self, time_seconds):
        """
        Get the sleep state at a specific time.
        
        Args:
            time_seconds: Time in seconds from start
            
        Returns:
            tuple: (state_code, state_name) or (None, None) if no data
        """
        if self.sleep_scoring_data is None:
            return None, None
        
        states = self.sleep_scoring_data['states']
        fs = self.sleep_scoring_data.get('sampling_rate', 1000)
        state_names = self.sleep_scoring_data.get('state_names', STATE_NAMES)
        
        # Convert time to sample index
        idx = int(time_seconds * fs)
        
        if 0 <= idx < len(states):
            state = states[idx]
            name = state_names.get(state, f'State {state}')
            return state, name
        
        return None, None