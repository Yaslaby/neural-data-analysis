"""
Sleep Scoring Dialog for Loading and Processing .mat Files
Handles loading of MATLAB sleep scoring files and alignment with neural data
WITH COLOR LEGEND DISPLAY
"""

import os
import numpy as np
from scipy.io import loadmat
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QMessageBox, QGroupBox,
                             QGridLayout, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QComboBox, QDialogButtonBox, QTextEdit,
                             QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

class ColoredSquare(QFrame):
    """Widget to display a colored square for legend"""
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color[0], color[1], color[2], color[3]))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

class SleepScoringDialog(QDialog):
    """Dialog for loading and configuring sleep scoring data"""
    
    # Color scheme for sleep states (RGBA format) - Pastel & Faded
    STATE_COLORS = {
        0: (220, 220, 220, 100),  # Unscored - Light Grey
        1: (200, 230, 200, 120),  # Awake - Soft Green
        3: (200, 220, 240, 120),  # Non-REM - Soft Blue
        5: (240, 210, 220, 120),  # REM - Soft Pink
        4: (230, 210, 240, 120)   # Intermediate - Soft Purple
    }
    STATE_NAMES = {
        0: 'Unscored/Unknown',
        1: 'Awake',
        3: 'Non-REM Sleep',
        5: 'REM (Rapid Eye Movement)',
        4: 'Intermediate Stage'
    }
    
    def __init__(self, neural_data_duration=None, neural_fs=None, parent=None):
        super().__init__(parent)
        self.neural_data_duration = neural_data_duration  # Duration in seconds
        self.neural_fs = neural_fs  # Sampling rate of neural data
        self.mat_file_path = None
        self.states_data = None
        
        self.setWindowTitle("Load Sleep Scoring Data")
        self.setModal(True)
        self.resize(700, 750)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Load Sleep Scoring from .mat File")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 10px; background: #f0f8ff; border: 1px solid #ccc;")
        layout.addWidget(title)
        
        # Color legend (MODIFIED - RGB removed, 12pt font)
        self.setup_color_legend(layout)
        
        # File selection
        self.setup_file_selection(layout)
        
        # Neural data info
        if self.neural_data_duration:
            self.setup_neural_info(layout)
        
        # Configuration
        self.setup_configuration(layout)
        
        # Preview
        self.setup_preview(layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Load & Align")
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(buttons)
    
    def setup_color_legend(self, layout):
        """Setup color legend display - MODIFIED: RGB removed, 12pt font"""
        group = QGroupBox("Sleep State Color Coding")
        group_layout = QGridLayout(group)
        group_layout.setSpacing(10)
        
        info_label = QLabel("These colors will be used to display sleep states in the comparison view:")
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Arial", 12))
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        group_layout.addWidget(info_label, 0, 0, 1, 3)
        
        # Display each state with its color
        for state_code in sorted(self.STATE_COLORS.keys()):
            color = self.STATE_COLORS[state_code]
            name = self.STATE_NAMES[state_code]
            
            row = state_code + 1
            
            # State code label
            code_label = QLabel(f"State {state_code}:")
            code_label.setFont(QFont("Arial", 12, QFont.Bold))
            group_layout.addWidget(code_label, row, 0)
            
            # Color square
            color_square = ColoredSquare(color)
            group_layout.addWidget(color_square, row, 1)
            
            # State name
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 12))
            name_label.setStyleSheet("padding: 5px;")
            group_layout.addWidget(name_label, row, 2)
        
        layout.addWidget(group)
    
    def setup_file_selection(self, layout):
        """Setup file selection group"""
        group = QGroupBox("Sleep Scoring File (.mat)")
        group_layout = QVBoxLayout(group)
        
        # File path display
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 3px;
                font-family: monospace;
            }
        """)
        file_layout.addWidget(self.file_path_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        group_layout.addLayout(file_layout)
        
        # Instructions
        instructions = QLabel(
            "Select a MATLAB .mat file containing sleep scoring data.\n"
            "The file should contain a 'states' variable with sleep state codes as shown above."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        group_layout.addWidget(instructions)
        
        layout.addWidget(group)
    
    def setup_neural_info(self, layout):
        """Show neural data information"""
        group = QGroupBox("Neural Data Information")
        group_layout = QGridLayout(group)
        
        duration_label = QLabel(f"Duration: {self.neural_data_duration:.2f} seconds "
                               f"({self.neural_data_duration/60:.1f} minutes)")
        duration_label.setStyleSheet("font-weight: bold; color: #0078d4;")
        group_layout.addWidget(QLabel("Neural Data:"), 0, 0)
        group_layout.addWidget(duration_label, 0, 1)
        
        if self.neural_fs:
            fs_label = QLabel(f"{self.neural_fs:.0f} Hz")
            fs_label.setStyleSheet("font-weight: bold; color: #0078d4;")
            group_layout.addWidget(QLabel("Sampling Rate:"), 1, 0)
            group_layout.addWidget(fs_label, 1, 1)
        
        layout.addWidget(group)
    
    def setup_configuration(self, layout):
        """Setup configuration options"""
        group = QGroupBox("Configuration")
        group_layout = QGridLayout(group)
        
        # Time bin size (seconds per sleep state sample)
        group_layout.addWidget(QLabel("Time bin size (seconds):"), 0, 0)
        self.bin_size = QDoubleSpinBox()
        self.bin_size.setRange(0.1, 10.0)
        self.bin_size.setValue(1.0)
        self.bin_size.setSingleStep(0.1)
        self.bin_size.setToolTip("Duration in seconds of each sleep scoring bin (usually 1 second)")
        group_layout.addWidget(self.bin_size, 0, 1)
        
        # Alignment method
        group_layout.addWidget(QLabel("Alignment method:"), 1, 0)
        self.align_method = QComboBox()
        self.align_method.addItems([
            "Pad with zeros if shorter"
        ])
        self.align_method.setToolTip("Pad sleep scoring with zeros (unscored) if shorter than neural data")
        group_layout.addWidget(self.align_method, 1, 1)
        
        # Auto-detect bin size
        self.auto_detect = QCheckBox("Auto-detect bin size from file duration")
        self.auto_detect.setChecked(True)
        group_layout.addWidget(self.auto_detect, 2, 0, 1, 2)
        
        layout.addWidget(group)
    
    def setup_preview(self, layout):
        """Setup preview section"""
        group = QGroupBox("Preview & Validation")
        group_layout = QVBoxLayout(group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setStyleSheet("""
            QTextEdit {
                font-family: monospace;
                font-size: 9pt;
                background: #f9f9f9;
                border: 1px solid #ddd;
            }
        """)
        self.preview_text.setPlainText("Load a file to see preview...")
        group_layout.addWidget(self.preview_text)
        
        layout.addWidget(group)
    
    def browse_file(self):
        """Browse for .mat file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Sleep Scoring File",
            "",
            "MATLAB Files (*.mat);;All Files (*)"
        )
        
        if file_path:
            self.load_and_validate_file(file_path)
    
    def load_and_validate_file(self, file_path):
        """Load and validate the .mat file"""
        try:
            # Load file
            data = loadmat(file_path)
            
            # Check for 'states' variable
            if 'states' not in data:
                QMessageBox.warning(
                    self, "Invalid File",
                    "File does not contain a 'states' variable.\n\n"
                    "Expected structure: file.mat with 'states' array containing sleep state codes."
                )
                return
            
            # Extract and validate states
            states = data['states'].squeeze()
            
            if states.ndim != 1:
                QMessageBox.warning(
                    self, "Invalid Format",
                    f"'states' variable has unexpected shape: {states.shape}\n"
                    "Expected 1D array of sleep state codes."
                )
                return
            
            # Store data
            self.mat_file_path = file_path
            self.states_data = states
            
            # Update UI
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setStyleSheet("""
                QLabel {
                    padding: 8px;
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 3px;
                    font-family: monospace;
                    color: #155724;
                }
            """)
            
            # Auto-detect bin size if enabled
            if self.auto_detect.isChecked() and self.neural_data_duration:
                detected_bin_size = self.neural_data_duration / len(states)
                self.bin_size.setValue(detected_bin_size)
            
            # Update preview
            self.update_preview()
            
            # Enable OK button
            self.ok_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load file:\n{str(e)}"
            )
    
    def update_preview(self):
        """Update the preview text with color indicators"""
        if self.states_data is None:
            return
        
        # Calculate statistics
        unique_states = np.unique(self.states_data)
        n_bins = len(self.states_data)
        bin_size = self.bin_size.value()
        total_duration = n_bins * bin_size
        
        # Build preview text
        preview = "FILE INFORMATION:\n"
        preview += "=" * 60 + "\n"
        preview += f"File: {os.path.basename(self.mat_file_path)}\n"
        preview += f"Number of time bins: {n_bins}\n"
        preview += f"Bin size: {bin_size} seconds\n"
        preview += f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)\n"
        preview += f"\nUnique states found: {unique_states}\n"
        
        preview += "\nSTATE DISTRIBUTION:\n"
        preview += "=" * 60 + "\n"
        
        for state in sorted(unique_states):
            count = np.sum(self.states_data == state)
            percentage = (count / n_bins) * 100
            duration = count * bin_size
            name = self.STATE_NAMES.get(state, f'State {state}')
            
            # Add color indicator
            if state == 0:
                color_name = "Lt Gray"
            elif state == 1:
                color_name = "Green"
            elif state == 3:
                color_name = "Blue"
            elif state == 4:
                color_name = "Purple"
            elif state == 5:
                color_name = "Pink"
            else:
                color_name = "Unknown"
            
            preview += f"{name:30s} [{color_name:6s}]:\n"
            preview += f"  {count:6d} bins ({percentage:5.1f}%) = {duration:7.1f}s ({duration/60:.1f}min)\n"
        
        # Alignment check
        if self.neural_data_duration:
            preview += "\nALIGNMENT CHECK:\n"
            preview += "=" * 60 + "\n"
            preview += f"Neural data duration:   {self.neural_data_duration:.2f} seconds\n"
            preview += f"Sleep scoring duration: {total_duration:.2f} seconds\n"
            
            diff = abs(self.neural_data_duration - total_duration)
            if diff < 5:  # Within 5 seconds
                preview += f"âœ… Durations match well (difference: {diff:.2f}s)\n"
            elif diff < 60:  # Within 1 minute
                preview += f"âš ï¸  Durations differ by {diff:.2f}s (may need adjustment)\n"
            else:
                preview += f"âŒ Large duration difference: {diff:.2f}s\n"
                preview += "   Consider checking your data or bin size settings.\n"
        
        self.preview_text.setPlainText(preview)
    
    def get_sleep_scoring_data(self):
        """Get processed sleep scoring data aligned with neural data"""
        if self.states_data is None:
            return None
        
        bin_size = self.bin_size.value()
        
        # Upsample to match neural sampling rate
        if self.neural_fs:
            # Each sleep state bin represents bin_size seconds
            # Repeat each state for (bin_size * neural_fs) samples
            samples_per_bin = int(bin_size * self.neural_fs)
            scoring_upsampled = np.repeat(self.states_data, samples_per_bin)
            
            # Align with neural data length
            target_len = int(self.neural_data_duration * self.neural_fs)
            current_len = len(scoring_upsampled)
            
            if current_len > target_len:
                # Trim
                scoring_upsampled = scoring_upsampled[:target_len]
            elif current_len < target_len:
                # Pad
                scoring_upsampled = np.pad(
                    scoring_upsampled,
                    (0, target_len - current_len),
                    mode='constant',
                    constant_values=0
                )
            
            return {
                'states': scoring_upsampled,
                'original_states': self.states_data,
                'bin_size': bin_size,
                'file_path': self.mat_file_path,
                'sampling_rate': self.neural_fs,
                'state_names': self.STATE_NAMES,
                'state_colors': self.STATE_COLORS  # Include colors in output
            }
        else:
            # Return original if no neural_fs provided
            return {
                'states': self.states_data,
                'original_states': self.states_data,
                'bin_size': bin_size,
                'file_path': self.mat_file_path,
                'state_names': self.STATE_NAMES,
                'state_colors': self.STATE_COLORS  # Include colors in output
            }


# Test function
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Test with example parameters
    dialog = SleepScoringDialog(
        neural_data_duration=1800.0,  # 30 minutes
        neural_fs=1000.0  # 1000 Hz
    )
    
    if dialog.exec_() == QDialog.Accepted:
        data = dialog.get_sleep_scoring_data()
        if data:
            print("âœ… Sleep scoring loaded successfully!")
            print(f"   States shape: {data['states'].shape}")
            print(f"   Unique states: {np.unique(data['states'])}")
            print(f"   Colors: {data['state_colors']}")
    
    sys.exit()