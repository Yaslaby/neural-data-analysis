"""
Comparison View Module
Handles the before/after preprocessing comparison visualization
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QToolBar,
    QPushButton, QComboBox, QLabel, QSlider, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from annotation import AnnotationControls, AnnotationWidget, add_drag_to_mark_annotation
from PyQt5.QtWidgets import QMessageBox

class comparison_view:
    """Class containing all comparison view methods for OpenEphysMainWindow"""
    def create_comparison_view(self, raw_data, raw_timestamps, proc_data, proc_timestamps, 
                          channel_names, original_fs, target_fs, raw_thresholds=None, proc_thresholds=None):
        """Create comparison view with amplitude control support and resizable splitter"""
        
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
            colors = ['black', '#ff7f0e', '#2ca02c', '#d62728']
            
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
                pen = pg.mkPen(color=colors[i % len(colors)], width=1)
                
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
            self.comparison_proc_threshold_lines = []
            self.comparison_proc_threshold_base = []
            for i in range(n_channels):
                y_offset = (n_channels - 1 - i) * self.comparison_spacing
                
                ch_data = proc_data[:, i]
                normalized_base = (ch_data / np.ptp(ch_data)) * self.comparison_spacing * 0.8
                self.comparison_proc_normalized.append(normalized_base)
                
                normalized = normalized_base + y_offset
                pen = pg.mkPen(color=colors[i % len(colors)], width=1, style=Qt.SolidLine)
                
                curve = self.comparison_plot_widget.plot(
                    proc_timestamps, normalized,
                    pen=pen,
                    antialias=False,
                    clipToView=True,
                    autoDownsample=True,
                    downsampleMethod='peak'
                )
                self.comparison_proc_curves.append(curve)
                
                # ADD THRESHOLD LINE FOR AFTER (processed)
                if proc_thresholds and i < len(proc_thresholds):
                    threshold_value = proc_thresholds[i]
                    
                    from scipy.signal import hilbert
                    envelope = np.abs(hilbert(ch_data))
                    envelope_ptp = np.ptp(envelope)
                    
                    if envelope_ptp > 0:
                        # Store base threshold (without offset) for amplitude scaling
                        threshold_base = (threshold_value / envelope_ptp) * self.comparison_spacing * 0.8
                        threshold_normalized = threshold_base + y_offset
                        
                        pen_thresh = pg.mkPen(color='red', width=1, style=Qt.SolidLine)
                        thresh_line = self.comparison_plot_widget.plot(
                            [proc_timestamps[0], proc_timestamps[-1]], 
                            [threshold_normalized, threshold_normalized],
                            pen=pen_thresh
                        )
                        
                        # Store for amplitude updates
                        self.comparison_proc_threshold_lines.append(thresh_line)
                        self.comparison_proc_threshold_base.append(threshold_base)
                        
                        text_label = pg.TextItem(
                            text=f"Thresh (2.5σ): {threshold_value:.2f}",
                            color=(255, 0, 0),
                            anchor=(0, 0.5),
                            fill=(255, 255, 255, 200)
                        )
                        text_label.setPos(proc_timestamps[0] + (proc_timestamps[-1] - proc_timestamps[0]) * 0.02, threshold_normalized + self.comparison_spacing * 0.15)
                        self.comparison_plot_widget.addItem(text_label)

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
            print(" Comparison view created with RESIZABLE splitter!")
            
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
        
        print("setup_comparison_controls COMPLETE")

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
        
        print("Slider added to correct location (below plot, above annotations)")

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
        # Update threshold lines with same amplitude scale
        if hasattr(self, 'comparison_proc_threshold_lines') and hasattr(self, 'comparison_proc_threshold_base'):
            for i, (thresh_line, thresh_base) in enumerate(zip(self.comparison_proc_threshold_lines, self.comparison_proc_threshold_base)):
                y_offset = (n_channels - 1 - i) * self.comparison_spacing
                scaled_threshold = thresh_base * self.comparison_amp_scale + y_offset
                thresh_line.setData(
                    [self.comparison_proc_timestamps[0], self.comparison_proc_timestamps[-1]],
                    [scaled_threshold, scaled_threshold]
                )    

    def _back_to_data_view(self):
        """Return to main data view"""
        self.content_stack.setCurrentWidget(self.plot_page)
        
        if hasattr(self, 'comparison_toolbar'):
            self.removeToolBar(self.comparison_toolbar)
        
        if hasattr(self, 'plot_manager'):
            self.plot_data()