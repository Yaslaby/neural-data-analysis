import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QToolBar, QAction, QComboBox, QLabel, QPushButton, 
                             QSlider, QVBoxLayout, QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt
import os
from pathlib import Path

class PlotManager:
    def __init__(self, parent):
        self.parent = parent
        
        if hasattr(parent, 'plot_widget') and parent.plot_widget:
            self.plot_widget = parent.plot_widget
        else:
            print("ERROR: no plot widget!")
            self.plot_widget = None
            return
        
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.full_data = None
        self.absolute_timestamps = None
        self.time_range = 0.0
        self.start_time = 0.0
        self.sample_rate = 30000.0
        self.file_name = ""
        self.is_multichannel = False
        self.channel_names = []
        
        self.nav_slider = None
        self.slider_updating = False
        self.view_updating = False
        
        self.amplitude_scales = []
        self.offset_step = 10
        self.offsets = []
        self.selected_channel = 0
        
        self.plot_curves = []
        self.base_scaled_data = []
        
        self.time_windows = [
            ("Full View", None),
            ("10 min", 600), ("5 min", 300), ("1 min", 60),
            ("30 sec", 30), ("10 sec", 10), ("1 sec", 1),
            ("100 ms", 0.1), ("10 ms", 0.01)
        ]
        self.current_window_index = 0
        
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def plot_data(self, data, timestamps, header, file_path):
        if not self.plot_widget:
            return
        
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.getPlotItem().setDownsampling(auto=True, mode='peak')
        
        if timestamps is not None and len(timestamps) > 0:
        # Ensure timestamps start at zero
            if timestamps[0] != 0:
                self.zero_based_timestamps = timestamps - timestamps[0]
            else:
                self.zero_based_timestamps = timestamps.copy()
            self.start_time = 0.0  # Always start at zero
            self.time_range = self.zero_based_timestamps[-1]
        else:
            self.sample_rate = header.get('sampleRate', 30000.0)
            self.zero_based_timestamps = np.arange(len(data)) / self.sample_rate
            self.start_time = 0.0
            self.time_range = self.zero_based_timestamps[-1]
        
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            data = data.reshape(-1, 1)
            self.is_multichannel = False
            
            if hasattr(self, 'parent') and hasattr(self.parent, 'datasets') and self.parent.current_index >= 0:
                dataset_name = self.parent.datasets[self.parent.current_index]["name"]
                self.channel_names = [dataset_name]
            else:
                self.channel_names = [os.path.splitext(os.path.basename(str(file_path)))[0]]
        else:
            self.is_multichannel = True
            n_ch = data.shape[1]
            if 'channel_files' in header and header['channel_files']:
                self.channel_names = []
                for filename in header['channel_files']:
                    clean_name = os.path.splitext(filename)[0]
                    if clean_name.replace('_', '').replace('CH', '').isdigit():
                        parts = clean_name.split('_')
                        if len(parts) == 2 and parts[1].startswith('CH'):
                            clean_name = f"{parts[1]} ({parts[0]})"
                    self.channel_names.append(clean_name)
            else:
                self.channel_names = [f'CH{i+1}' for i in range(n_ch)]
        
        n_channels = data.shape[1]
        self.amplitude_scales = [1.0] * n_channels
        self.selected_channel = -1
        
        self.full_data = data
        self.sample_rate = header.get('sampleRate', 30000.0)
        self.file_name = os.path.basename(str(file_path))
        
        self.plot_widget.clear()
        self.plot_curves = []
        self.base_scaled_data = []
        
        self._configure_time_axis()
        
        if self.is_multichannel:
            self._plot_multi(data)
        else:
            self._plot_single(data.flatten())
        
        self.plot_widget.setLabel('bottom', 'Time (seconds)', units='')
        self.plot_widget.autoRange()
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        self._add_zoom_toolbar()
        self._add_slider()
        
        self.plot_widget.sigRangeChanged.connect(self._view_changed)
        
        self._setup_initial_ticks()
        
        print("plot ready")
    
    def _configure_time_axis(self):
        """Simple, reliable time axis configuration"""
        axis = self.plot_widget.getAxis('bottom')
        
        axis.enableAutoSIPrefix(False)
        axis.setLabel('Time (seconds)', units='')
        
        self.time_axis = axis
        
        axis.setStyle(
            tickTextOffset=10,
            tickLength=6,
            autoReduceTextSpace=False
        )
    
    def _update_time_ticks(self):
        """FIXED: Complete tick labeling with better handling of small windows"""
        if not hasattr(self, 'time_axis') or self.zero_based_timestamps is None:
            return
            
        try:
            view_range = self.plot_widget.viewRange()[0]
            start_time = view_range[0]
            end_time = view_range[1]
            duration = end_time - start_time
            
            # FIXED: Better spacing for very small time windows
            if duration <= 0.001:    # 1ms or less
                spacing = 0.0001     # Every 0.1ms (100 microseconds)
            elif duration <= 0.005:  # 5ms or less
                spacing = 0.0005     # Every 0.5ms
            elif duration <= 0.01:   # 10ms or less
                spacing = 0.001      # Every 1ms  
            elif duration <= 0.02:   # 20ms or less
                spacing = 0.002      # Every 2ms
            elif duration <= 0.05:   # 50ms or less
                spacing = 0.005      # Every 5ms
            elif duration <= 0.1:    # 100ms or less
                spacing = 0.01       # Every 10ms
            elif duration <= 0.2:    # 200ms or less  
                spacing = 0.02       # Every 20ms
            elif duration <= 0.5:    # 500ms or less
                spacing = 0.05       # Every 50ms
            elif duration <= 1:      # 1 second or less
                spacing = 0.1        # Every 100ms
            elif duration <= 2:      # 2 seconds or less
                spacing = 0.2        # Every 200ms
            elif duration <= 5:      # 5 seconds or less
                spacing = 0.5        # Every 500ms
            elif duration <= 10:     # 10 seconds or less
                spacing = 1          # Every 1 second
            elif duration <= 30:     # 30 seconds or less
                spacing = 5          # Every 5 seconds
            elif duration <= 60:     # 1 minute or less
                spacing = 10         # Every 10 seconds
            elif duration <= 300:    # 5 minutes or less
                spacing = 30         # Every 30 seconds
            elif duration <= 900:    # 15 minutes or less
                spacing = 60         # Every 1 minute
            else:                    # Very zoomed out
                spacing = 300        # Every 5 minutes
            
            # Generate ticks with proper coverage
            margin = spacing * 2
            extended_start = start_time - margin
            extended_end = end_time + margin
            
            start_tick = np.floor(extended_start / spacing) * spacing
            
            tick_positions = []
            tick_labels = []
            
            current_pos = start_tick
            while current_pos <= extended_end:
                tick_positions.append(current_pos)
                
                # Format labels with appropriate precision
                if spacing >= 1:
                    label = f"{int(round(current_pos))}"      
                elif spacing >= 0.1:
                    label = f"{current_pos:.1f}"              
                elif spacing >= 0.01:
                    label = f"{current_pos:.2f}"              
                elif spacing >= 0.001:
                    label = f"{current_pos:.3f}"
                else:
                    label = f"{current_pos:.4f}"  # For sub-millisecond
                
                tick_labels.append(label)
                current_pos += spacing
            
            # Apply ticks
            if tick_positions:
                major_ticks = list(zip(tick_positions, tick_labels))
                
                # Minor ticks for grid
                minor_spacing = spacing / 4
                minor_positions = []
                
                current_pos = start_tick
                while current_pos <= extended_end:
                    is_major = any(abs(current_pos - major_pos) < spacing * 0.01 
                                for major_pos, _ in major_ticks)
                    if not is_major:
                        minor_positions.append(current_pos)
                    current_pos += minor_spacing
                
                minor_ticks = [(pos, "") for pos in minor_positions]
                
                self.time_axis.setTicks([major_ticks, minor_ticks])
                
                print(f"Zoom: {duration*1000:.2f}ms, Spacing: {spacing*1000:.3f}ms, Ticks: {len(tick_positions)}")
        
        except Exception as e:
            print(f"Error updating ticks: {e}")
    
    def _setup_initial_ticks(self):
        """Setup initial ticks after plotting"""
        if hasattr(self, 'time_axis'):
            self.time_axis.enableAutoSIPrefix(False)
            self._update_time_ticks()
            print("Initial ticks configured")
    
    def _plot_single(self, data):
        pen = pg.mkPen(self.colors[0], width=1.5)
        
        opts = {
            'pen': pen,
            'antialias': False,
            'autoDownsample': True,
            'downsampleMethod': 'peak',
            'clipToView': True
        }
        
        curve = self.plot_widget.plot(self.zero_based_timestamps, data, **opts)
        self.plot_curves = [curve]
        self.plot_widget.setTitle("Single-channel upload (1 Channel)")
        # Set Y-axis to show channel name
        y_axis = self.plot_widget.getAxis('left')
        y_axis.setTicks([])
        y_axis.setLabel(self.channel_names[0], units='')  
    
    def _plot_multi(self, data):
        n_ch = min(8, data.shape[1])
        
        self.offset_step = self._calc_spacing(data[:, :n_ch])
        self.offsets = np.arange(n_ch) * self.offset_step
        
        self.base_scaled_data = []
        
        for i in range(n_ch):
            color = self.colors[i % len(self.colors)]
            pen = pg.mkPen(color, width=1.5)
            
            ch_data = data[:, i]
            ptp = np.ptp(ch_data)
            if ptp == 0:
                ptp = 1
            
            base_scaled = (ch_data / ptp) * self.offset_step * 0.8
            self.base_scaled_data.append(base_scaled)
            
            scaled_data = base_scaled * self.amplitude_scales[i] + self.offsets[i]
            
            opts = {
                'pen': pen,
                'name': self.channel_names[i],
                'antialias': False,
                'autoDownsample': True,
                'downsampleMethod': 'peak',
                'clipToView': True
            }
            
            curve = self.plot_widget.plot(self.zero_based_timestamps, scaled_data, **opts)
            self.plot_curves.append(curve)
            
        
        ticks = [(offset, name) for offset, name in zip(self.offsets, self.channel_names[:n_ch])]
        axis = self.plot_widget.getAxis('left')
        axis.setTicks([ticks])
        axis.setLabel('')
        
        self.plot_widget.setYRange(-self.offset_step, self.offsets[-1] + self.offset_step, padding=0.1)
        self.plot_widget.setTitle(f"Multi-channel upload ({n_ch} Channels)")
    
    def _calc_spacing(self, data):
        rms = [np.sqrt(np.mean(data[:, i]**2)) for i in range(data.shape[1])]
        base_spacing = max(3 * max(rms), 100)
        return max(base_spacing, 10)
    
    def _add_slider(self):
        layout = self.parent.plot_page.layout()
        if not layout:
            layout = QVBoxLayout(self.parent.plot_page)
        
        if hasattr(self, 'nav_slider') and self.nav_slider:
            self.nav_slider.setParent(None)
            self.nav_slider = None
        
        if hasattr(self, 'slider_widget') and self.slider_widget:
            self.slider_widget.setParent(None)
            self.slider_widget = None
        
        plot_idx = -1
        for i in range(layout.count()):
            if layout.itemAt(i).widget() == self.plot_widget:
                plot_idx = i
                break
        
        self.nav_slider = QSlider(Qt.Horizontal)
        self.nav_slider.setMinimum(0)
        
        if self.absolute_timestamps is not None and self.time_range > 0:
            self.nav_slider.setMaximum(1000)
            self.nav_slider.setValue(0)
        else:
            self.nav_slider.setMaximum(100)
            self.nav_slider.setValue(0)
        
        self.nav_slider.valueChanged.connect(self._slider_moved)
        
        self.nav_slider.setStyleSheet("""
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
        
        self.nav_slider.setMaximumHeight(30)
        
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        
        btn_style = """
            QPushButton {
                min-width: 30px;
                max-width: 30px;
                min-height: 25px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: #f8f8f8;
                margin: 0px 5px;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
        """
        
        self.btn_left.setStyleSheet(btn_style)
        self.btn_right.setStyleSheet(btn_style)
        
        self.btn_left.clicked.connect(self._move_left)
        self.btn_right.clicked.connect(self._move_right)
        
        self.slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.slider_widget)
        slider_layout.addWidget(self.btn_left)
        slider_layout.addWidget(self.nav_slider)
        slider_layout.addWidget(self.btn_right)
        
        if plot_idx >= 0:
            layout.insertWidget(plot_idx + 1, self.slider_widget)
        else:
            layout.addWidget(self.slider_widget)
    
    def _move_left(self):
        if self.absolute_timestamps is None:
            return
        val = self.nav_slider.value()
        step = max(1, int(self.nav_slider.maximum() / 50))
        new_val = max(val - step, 0)
        self.nav_slider.setValue(new_val)
    
    def _move_right(self):
        if self.absolute_timestamps is None:
            return
        val = self.nav_slider.value()
        step = max(1, int(self.nav_slider.maximum() / 50))
        new_val = min(val + step, self.nav_slider.maximum())
        self.nav_slider.setValue(new_val)
    
    def _slider_moved(self, val):
        """FIXED: Slider movement with guaranteed tick updates"""
        if self.slider_updating or self.zero_based_timestamps is None or self.time_range <= 0:
            return
        
        if self.view_updating:
            return
            
        self.view_updating = True
        
        try:
            relative_pos = val / self.nav_slider.maximum()
            absolute_time = self.start_time + relative_pos * self.time_range
            
            xrange = self.plot_widget.viewRange()[0]
            width = xrange[1] - xrange[0]
            
            start = absolute_time - width/2
            end = absolute_time + width/2
            
            end_time = self.start_time + self.time_range
            if start < self.start_time:
                start = self.start_time
                end = start + width
            elif end > end_time:
                end = end_time
                start = max(self.start_time, end - width)
            
            # Update view - temporarily disconnect to prevent recursion
            self.plot_widget.sigRangeChanged.disconnect()
            self.plot_widget.setXRange(start, end, padding=0)
            
            # CRITICAL: Force immediate tick update
            self._update_time_ticks()
            
            # Reconnect signal
            self.plot_widget.sigRangeChanged.connect(self._view_changed)
            
            print(f"Slider moved to: {start:.3f}-{end:.3f}s")
                
        finally:
            self.view_updating = False
    
    def _add_zoom_toolbar(self):
        if hasattr(self.parent, 'zoom_toolbar'):
            self.parent.removeToolBar(self.parent.zoom_toolbar)
        
        toolbar = QToolBar("Zoom")
        toolbar.setMovable(False)
        self.parent.addToolBar(Qt.TopToolBarArea, toolbar)
        self.parent.zoom_toolbar = toolbar
        
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
        
        btn = QPushButton("🔍+ Zoom In")
        btn.clicked.connect(self._zoom_in_synchronized)
        toolbar.addWidget(btn)
        
        btn = QPushButton("🔍- Zoom Out")
        btn.clicked.connect(self._zoom_out_synchronized)
        toolbar.addWidget(btn)
        
        btn = QPushButton("Reset")
        btn.clicked.connect(self._reset)
        toolbar.addWidget(btn)
        
        toolbar.addSeparator()
        
        if self.is_multichannel:
            toolbar.addWidget(QLabel("Channel:"))
            self.channel_combo = QComboBox()
            channel_items = ["All Channels"] + self.channel_names
            self.channel_combo.addItems(channel_items)
            self.channel_combo.currentIndexChanged.connect(self._channel_selected)
            toolbar.addWidget(self.channel_combo)
            
            btn_amp_in = QPushButton("+ Amp")
            btn_amp_in.clicked.connect(self._zoom_in_amplitude)
            toolbar.addWidget(btn_amp_in)
            
            btn_amp_out = QPushButton("- Amp")
            btn_amp_out.clicked.connect(self._zoom_out_amplitude)
            toolbar.addWidget(btn_amp_out)
            
            self.amp_label = QLabel("Amp: All 1.0x")
            self.amp_label.setStyleSheet("""
                QLabel {
                    color: #333;
                    padding: 4px 8px;
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 3px;
                }
            """)
            toolbar.addWidget(self.amp_label)
            
            toolbar.addSeparator()
        
        toolbar.addWidget(QLabel("Time Window:"))
        
        self.time_combo = QComboBox()
        self.time_combo.addItems([label for label, _ in self.time_windows])
        self.time_combo.currentTextChanged.connect(self._time_window_changed)
        toolbar.addWidget(self.time_combo)
        
        toolbar.addSeparator()
        
        self.view_label = QLabel("View: Full")
        self.view_label.setStyleSheet("""
            QLabel {
                color: #333;
                padding: 4px 8px;
                background: #e8f4fd;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        toolbar.addWidget(self.view_label)
    
    def _channel_selected(self, index):
        self.selected_channel = index - 1
        self._update_amplitude_label()
    
    def _update_amplitude_label(self):
        if not hasattr(self, 'amp_label'):
            return
            
        if self.selected_channel == -1:
            if len(self.amplitude_scales) > 0:
                avg_amp = sum(self.amplitude_scales) / len(self.amplitude_scales)
                self.amp_label.setText(f"Amp: All {avg_amp:.1f}x")
            else:
                self.amp_label.setText("Amp: All 1.0x")
        else:
            if self.selected_channel < len(self.amplitude_scales) and self.selected_channel < len(self.channel_names):
                channel_name = self.channel_names[self.selected_channel]
                self.amp_label.setText(f"Amp: {channel_name} {self.amplitude_scales[self.selected_channel]:.1f}x")
            else:
                self.amp_label.setText("Amp: 1.0x")
    
    def _zoom_in_synchronized(self):
        """Zoom in using predefined time window steps - FIXED"""
        if self.current_window_index < len(self.time_windows) - 1:
            self.current_window_index += 1
            
            # Disconnect combo to prevent recursion
            self.time_combo.currentTextChanged.disconnect()
            self.time_combo.setCurrentIndex(self.current_window_index)
            self.time_combo.currentTextChanged.connect(self._time_window_changed)
            
            # Apply the zoom
            self._apply_time_window_by_index(self.current_window_index)
    
    def _zoom_out_synchronized(self):
        """Zoom out using predefined time window steps - FIXED"""
        if self.current_window_index > 0:
            self.current_window_index -= 1
            
            # Disconnect combo to prevent recursion
            self.time_combo.currentTextChanged.disconnect()
            self.time_combo.setCurrentIndex(self.current_window_index)
            self.time_combo.currentTextChanged.connect(self._time_window_changed)
            
            # Apply the zoom
            self._apply_time_window_by_index(self.current_window_index)
    
    def _apply_time_window_by_index(self, index):
        """Apply time window by index - FIXED for proper zoom and tick updates"""
        if self.zero_based_timestamps is None or self.time_range <= 0:
            return
        
        label, duration = self.time_windows[index]
        
        # Block signal updates during this operation
        self.plot_widget.sigRangeChanged.disconnect()
        
        try:
            if label == "Full View":
                # Show full data range
                end_time = self.start_time + self.time_range
                self.plot_widget.setXRange(self.start_time, end_time, padding=0.02)
            else:
                # Get current view range
                xrange = self.plot_widget.viewRange()[0]
                current_center = (xrange[0] + xrange[1]) / 2
                current_width = xrange[1] - xrange[0]
                
                # FIXED: Better centering logic
                # If we're at full view or significantly different width, center on middle of data
                if abs(current_width - self.time_range) < self.time_range * 0.1:
                    # Currently at full view - center on data middle
                    current_center = self.start_time + self.time_range / 2
                elif current_width > duration * 5:
                    # Zooming in significantly - keep current center if valid
                    pass
                # Otherwise keep current center
                
                # Calculate new range
                start = current_center - duration/2
                end = current_center + duration/2
                
                # Ensure we stay within data bounds
                end_time = self.start_time + self.time_range
                if start < self.start_time:
                    start = self.start_time
                    end = min(start + duration, end_time)
                elif end > end_time:
                    end = end_time
                    start = max(self.start_time, end - duration)
                
                # Apply the range
                self.plot_widget.setXRange(start, end, padding=0)
            
            # Force immediate tick update
            self._update_time_ticks()
            
            # Update view label
            self._update_view_label()
            
            # Force a repaint
            self.plot_widget.update()
            
        finally:
            # Reconnect the signal
            self.plot_widget.sigRangeChanged.connect(self._view_changed)

    def _zoom_in_amplitude(self):
        if not self.is_multichannel:
            return
            
        if self.selected_channel == -1:
            for i in range(len(self.amplitude_scales)):
                self.amplitude_scales[i] *= 1.2
        else:
            if self.selected_channel < len(self.amplitude_scales):
                self.amplitude_scales[self.selected_channel] *= 1.2
        
        self._update_amplitude_fast()
        self._update_amplitude_label()
    
    def _zoom_out_amplitude(self):
        if not self.is_multichannel:
            return
            
        if self.selected_channel == -1:
            for i in range(len(self.amplitude_scales)):
                self.amplitude_scales[i] /= 1.2
        else:
            if self.selected_channel < len(self.amplitude_scales):
                self.amplitude_scales[self.selected_channel] /= 1.2
        
        self._update_amplitude_fast()
        self._update_amplitude_label()
    
    def _update_amplitude_fast(self):
        if not self.is_multichannel or len(self.base_scaled_data) == 0 or len(self.plot_curves) == 0:
            return
        
        for i, (curve, base_data) in enumerate(zip(self.plot_curves, self.base_scaled_data)):
            if curve is not None and i < len(self.offsets) and i < len(self.amplitude_scales):
                new_data = base_data * self.amplitude_scales[i] + self.offsets[i]
                # FIXED: Use zero_based_timestamps instead of absolute_timestamps
                curve.setData(self.zero_based_timestamps, new_data)  
        
            # Update y-axis range if needed
            if len(self.offsets) > 0:
                margin = self.offset_step * 0.2
                y_min = -margin
                y_max = self.offsets[-1] + margin
                self.plot_widget.setYRange(y_min, y_max, padding=0.1)
    
    def _reset(self):
        if self.absolute_timestamps is not None and self.time_range > 0:
            end_time = self.start_time + self.time_range
            self.plot_widget.setXRange(self.start_time, end_time, padding=0.02)
        else:
            self.plot_widget.autoRange()
            
        self.current_window_index = 0
        self.time_combo.setCurrentText("Full View")
        
        if self.is_multichannel:
            self.amplitude_scales = [1.0] * len(self.amplitude_scales)
            self._update_amplitude_fast()
            self._update_amplitude_label()
        
        if self.nav_slider:
            self.nav_slider.setValue(0)
        
        self._update_view_label()
    
    def _time_window_changed(self, text):
        """Handle time window selection from combo box - FIXED"""
        if self.zero_based_timestamps is None:
            return
        
        # Find the index of the selected time window
        for i, (label, duration) in enumerate(self.time_windows):
            if label == text:
                self.current_window_index = i
                self._apply_time_window_by_index(i)
                break
    
    def _view_changed(self):
        """Enhanced view change handler with reliable tick updates"""
        if self.view_updating:
            return
        
        if hasattr(self, '_last_view_update'):
            import time
            if time.time() - self._last_view_update < 0.1:
                return
        
        try:
            # Always update ticks when view changes
            self._update_time_ticks()
            
            # Ensure SI prefixes stay disabled
            if hasattr(self, 'time_axis'):
                self.time_axis.enableAutoSIPrefix(False)
            
            self._update_view_label()
            self._sync_combo_with_current_view()
            
            # Sync slider
            if (hasattr(self, 'nav_slider') and self.nav_slider is not None and 
                not self.view_updating and self.zero_based_timestamps is not None and 
                self.time_range > 0):
                
                xrange = self.plot_widget.viewRange()[0]
                center = (xrange[0] + xrange[1]) / 2
                
                relative_pos = (center - self.start_time) / self.time_range
                relative_pos = max(0.0, min(1.0, relative_pos))
                slider_val = int(relative_pos * self.nav_slider.maximum())
                
                self.slider_updating = True
                self.nav_slider.setValue(slider_val)
                self.slider_updating = False
                
        finally:
            import time
            self._last_view_update = time.time()
    
    def _sync_combo_with_current_view(self):
        """Sync the time window combo box with current view when zooming with mouse"""
        if not hasattr(self, 'time_combo') or self.zero_based_timestamps is None:
            return
        
        xrange = self.plot_widget.viewRange()[0]
        current_duration = xrange[1] - xrange[0]
        
        best_match_index = 0
        min_diff = float('inf')
        
        for i, (label, duration) in enumerate(self.time_windows):
            if duration is None:  # Full view
                if abs(current_duration - self.time_range) < min_diff:
                    min_diff = abs(current_duration - self.time_range)
                    best_match_index = i
            else:
                if abs(current_duration - duration) < min_diff:
                    min_diff = abs(current_duration - duration)
                    best_match_index = i
        
        # FIXED: More lenient threshold - update if reasonably close (within 50%)
        if self.current_window_index != best_match_index and min_diff < current_duration * 0.5:
            self.current_window_index = best_match_index
            self.time_combo.currentTextChanged.disconnect()
            self.time_combo.setCurrentIndex(best_match_index)
            self.time_combo.currentTextChanged.connect(self._time_window_changed)
            print(f"Time window synced to: {self.time_windows[best_match_index][0]}")

    def _update_view_label(self):
        if not hasattr(self, 'view_label'):
            return
        
        xrange = self.plot_widget.viewRange()[0]
        dur = xrange[1] - xrange[0]
        start_time = xrange[0]
        
        start_seconds = int(round(start_time))
        end_seconds = int(round(xrange[1]))
        
        if dur >= 60:
            txt = f"View: {dur/60:.1f} min"
        elif dur >= 1:
            txt = f"View: {dur:.1f} sec"
        else:
            txt = f"View: {dur*1000:.1f} ms"
            
        txt += f" ({start_seconds}s-{end_seconds}s)"
        
        self.view_label.setText(txt)
    
    def reset_zoom(self):
        self._reset()