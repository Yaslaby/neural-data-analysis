"""
Sleep Scoring Integration
Handles sleep scoring visualization and legend in the comparison view
"""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QFrame,
    QMessageBox, QDialog
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt

# State color definitions
STATE_COLORS = {
    0: (220, 220, 220, 100),  # Unscored - Light Grey
    1: (200, 230, 200, 120),  # Awake - Soft Green
    3: (200, 220, 240, 120),  # Non-REM - Soft Blue
    5: (240, 210, 220, 120),  # REM - Soft Pink
    4: (230, 210, 240, 120)   # Intermediate - Soft Purple
}

STATE_NAMES = {
    0: 'Unscored',
    1: 'Awake',
    3: 'Non-REM',
    5: 'REM',
    4: 'Intermediate'
}

class sleep_scoring:
    """Mixin class containing sleep scoring methods for OpenEphysMainWindow"""

    def create_sleep_scoring_legend(self):
        """Create a compact sleep scoring color legend widget"""
        legend_widget = QWidget()
        legend_widget.setMaximumHeight(200)
        legend_layout = QVBoxLayout(legend_widget)
        legend_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Sleep State Color Legend")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 3px; background: #f0f8ff; border: 1px solid #ccc;")
        legend_layout.addWidget(title)

        # Color grid
        colors_group = QWidget()
        colors_layout = QGridLayout(colors_group)
        colors_layout.setSpacing(3)
        colors_layout.setContentsMargins(0, 0, 0, 0)

        row = 0
        for state_code in sorted(STATE_COLORS.keys()):
            color = STATE_COLORS[state_code]
            name = STATE_NAMES[state_code]

            # Color square
            color_square = QFrame()
            color_square.setFixedSize(20, 20)
            color_square.setFrameStyle(QFrame.Box | QFrame.Plain)
            color_square.setLineWidth(1)

            palette = color_square.palette()
            palette.setColor(QPalette.Window, QColor(color[0], color[1], color[2], color[3]))
            color_square.setPalette(palette)
            color_square.setAutoFillBackground(True)

            colors_layout.addWidget(color_square, row, 0)
            
            # State name
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 12))
            name_label.setStyleSheet("padding: 2px;")
            colors_layout.addWidget(name_label, row, 1)

            row += 1

        legend_layout.addWidget(colors_group)
        legend_layout.addStretch()

        return legend_widget

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
                    f"Sleep scoring will appear in comparison view when you preprocess data."
                )
    
    def add_sleep_scoring_to_comparison(self):
        """Add sleep scoring track to comparison view"""
        if self.sleep_scoring_data is None or not hasattr(self, 'comparison_plot_widget'):
            return

        try:
            states = self.sleep_scoring_data['states']
            state_names = self.sleep_scoring_data['state_names']

            # Create timestamps for sleep scoring
            if hasattr(self, 'comparison_raw_timestamps') and self.comparison_raw_timestamps is not None:
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.linspace(
                    self.comparison_raw_timestamps[0],
                    self.comparison_raw_timestamps[-1],
                    len(states)
                )
            else:
                fs = self.sleep_scoring_data.get('sampling_rate', 1000)
                timestamps = np.arange(len(states)) / fs
            
            # Calculate y-position for sleep scoring track
            n_channels = len(self.comparison_channel_names) if hasattr(self, 'comparison_channel_names') else 1
            spacing = self.comparison_spacing if hasattr(self, 'comparison_spacing') else 100

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
                if i == len(states) or states[i] != current_state:
                    start_time = timestamps[start_idx]
                    end_time = timestamps[i-1] if i < len(timestamps) else timestamps[-1]
                    
                    color = STATE_COLORS.get(current_state, (128, 128, 128, 100))
                    
                    from pyqtgraph import LinearRegionItem
                    region = LinearRegionItem(
                        values=[start_time, end_time],
                        orientation='vertical',
                        brush=pg.mkBrush(color),
                        movable=False,
                        pen=pg.mkPen(None)
                    )

                    region.setZValue(-10)
                    self.comparison_plot_widget.addItem(region)
                    self.sleep_scoring_items.append(region)

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

            for state_val, color in STATE_COLORS.items():
                if state_val in np.unique(states):
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

                    text = pg.TextItem(
                        state_names.get(state_val, f'State {state_val}'),
                        anchor=(0, 0.5),
                        color=(0, 0, 0)
                    )
                    text.setPos(legend_x + box_width * 1.5, legend_y)
                    self.comparison_plot_widget.addItem(text)
                    self.sleep_scoring_items.append(text)

                    legend_x += (timestamps[-1] - timestamps[0]) * 0.12

            # Update y-axis range to include sleep scoring
            current_y_range = self.comparison_plot_widget.viewRange()[1]
            new_max = max(current_y_range[1], track_y_position + track_height * 1.2)
            self.comparison_plot_widget.setYRange(
                current_y_range[0],
                new_max,
                padding=0.05
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()