"""
Annotation System for Manual Event Marking with Start/End Times
Supports ripples, spindles, delta waves with left-click drag-to-mark capability
Updated: Light grey color, left-click drag, simplified dialog
"""

import csv
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
                             QDialogButtonBox, QListWidget, QFileDialog, 
                             QMessageBox, QInputDialog, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import pyqtgraph as pg

class Annotation:
    """Annotation object with start and end times"""
    def __init__(self, start_time, end_time=None, description="", category="other", channel_name=""):
        self.start_time = float(start_time)
        # If no end_time provided, make it a point event (end = start)
        self.end_time = float(end_time) if end_time is not None else self.start_time
        self.description = str(description)
        self.category = str(category)
        self.channel_name = str(channel_name)
    
    @property
    def time(self):
        """Legacy property for backward compatibility - returns start_time"""
        return self.start_time
    
    @property
    def duration(self):
        """Calculate duration in seconds"""
        return self.end_time - self.start_time
    
    def is_point_event(self):
        """Check if this is a point event (no duration)"""
        return abs(self.end_time - self.start_time) < 0.001  # Less than 1ms
    
    def to_dict(self):
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration ms': self.duration * 1000,  # in milliseconds
            'description': self.description,
            'category': self.category,
            'channel_name': self.channel_name
        }

class AnnotationManager(QObject):
    """Annotation manager with start/end time support"""
    
    annotations_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.annotations = []
        # Removed 'manual' since all annotations are manual
        self.categories = ['ripple', 'spindle', 'delta', 'artifact']
    
    def add_annotation(self, start_time, end_time=None, description="", category="other", channel_name=""):
        """Add new annotation with start and end times"""
        annotation = Annotation(start_time, end_time, description, category, channel_name)
        self.annotations.append(annotation)
        self.annotations_changed.emit()
        return annotation
    
    def remove_annotation(self, index):
        """Remove annotation by index"""
        if 0 <= index < len(self.annotations):
            del self.annotations[index]
            self.annotations_changed.emit()
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations.clear()
        self.annotations_changed.emit()
    
    def get_annotations_by_category(self, category):
        """Get annotations of specific category"""
        return [ann for ann in self.annotations if ann.category == category]
    
    def export_to_csv(self, file_path):
        """Export annotations to CSV file with start/end times"""
        if not self.annotations:
            raise ValueError("No annotations to export")
        
        # Sort annotations by start time
        sorted_annotations = sorted(self.annotations, key=lambda x: x.start_time)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['start_time', 'end_time', 'duration ms', 'description', 'category', 'channel_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for ann in sorted_annotations:
                writer.writerow(ann.to_dict())

class AnnotationDialog(QDialog):
    """Dialog for adding/editing annotations with start and end times"""
    
    def __init__(self, annotation=None, categories=None, parent=None, current_time=0.0):
        super().__init__(parent)
        self.annotation = annotation
        self.categories = categories or ['ripple', 'spindle', 'delta', 'artifact']
        self.current_time = current_time
        
        self.setWindowTitle("Add Event" if annotation is None else "Edit Event")
        self.setModal(True)
        self.resize(400, 250)
        self.setup_ui()
        
        if annotation:
            self.load_annotation_data()
        else:
            # Set default times for new annotation
            self.start_time_edit.setValue(current_time)
            self.end_time_edit.setValue(current_time)
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Time settings group
        time_group = QGroupBox("Event Timing")
        time_layout = QVBoxLayout(time_group)
        
        # Start time
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Time (s):"))
        self.start_time_edit = QDoubleSpinBox()
        self.start_time_edit.setRange(0, 999999)
        self.start_time_edit.setDecimals(3)
        self.start_time_edit.valueChanged.connect(self.update_duration)
        start_layout.addWidget(self.start_time_edit)
        time_layout.addLayout(start_layout)
        
        # End time
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Time (s):"))
        self.end_time_edit = QDoubleSpinBox()
        self.end_time_edit.setRange(0, 999999)
        self.end_time_edit.setDecimals(3)
        self.end_time_edit.valueChanged.connect(self.update_duration)
        end_layout.addWidget(self.end_time_edit)
        time_layout.addLayout(end_layout)
        
        # Duration display
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration:"))
        self.duration_label = QLabel("0.000 s")
        self.duration_label.setStyleSheet("QLabel { color: #0078d4; font-weight: bold; }")
        duration_layout.addWidget(self.duration_label)
        duration_layout.addStretch()
        time_layout.addLayout(duration_layout)
        
        layout.addWidget(time_group)
        
        # Description
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Optional description of the event")
        desc_layout.addWidget(self.desc_edit)
        layout.addLayout(desc_layout)

        # Channel Name field
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel Name:"))
        self.channel_edit = QLineEdit()
        self.channel_edit.setPlaceholderText("e.g., CH1, LFP_1, etc.")
        channel_layout.addWidget(self.channel_edit)
        layout.addLayout(channel_layout)
        
        # Category
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.categories)
        cat_layout.addWidget(self.category_combo)
        layout.addLayout(cat_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Initial update
        self.update_duration()
    
    def update_duration(self):
        """Update the duration display"""
        start = self.start_time_edit.value()
        end = self.end_time_edit.value()
        duration = end - start
        
        if abs(duration) < 0.001:
            self.duration_label.setText("Point event")
            self.duration_label.setStyleSheet("QLabel { color: #666; font-weight: bold; }")
        elif duration < 0:
            self.duration_label.setText(f"{duration:.3f} s (Invalid!)")
            self.duration_label.setStyleSheet("QLabel { color: #d62728; font-weight: bold; }")
        elif duration < 1:
            self.duration_label.setText(f"{duration*1000:.1f} ms")
            self.duration_label.setStyleSheet("QLabel { color: #0078d4; font-weight: bold; }")
        else:
            self.duration_label.setText(f"{duration:.3f} s")
            self.duration_label.setStyleSheet("QLabel { color: #0078d4; font-weight: bold; }")
    
    def load_annotation_data(self):
        """Load existing annotation data"""
        if self.annotation:
            self.start_time_edit.setValue(self.annotation.start_time)
            self.end_time_edit.setValue(self.annotation.end_time)
            self.desc_edit.setText(self.annotation.description)
            self.channel_edit.setText(self.annotation.channel_name)
            self.category_combo.setCurrentText(self.annotation.category)
    
    def get_annotation_data(self):
        """Get annotation data from form"""
        start = self.start_time_edit.value()
        end = self.end_time_edit.value()
        
        # Validate that end >= start
        if end < start:
            QMessageBox.warning(self, "Invalid Time Range", 
                              "End time must be greater than or equal to start time.")
            return None
        
        return {
            'start_time': start,
            'end_time': end,
            'description': self.desc_edit.text(),
            'category': self.category_combo.currentText(),
            'channel_name': self.channel_edit.text()
        }

class AnnotationWidget(QListWidget):
    """Annotation list widget with start/end time display"""
    
    annotation_clicked = pyqtSignal(float)  # time to navigate to
    
    def __init__(self, annotation_manager, parent=None):
        super().__init__(parent)
        self.annotation_manager = annotation_manager
        
        # Setup UI
        self.setMaximumHeight(200)
        
        # Connect signals
        self.annotation_manager.annotations_changed.connect(self.refresh_annotations)
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # Context menu for right-click
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        self.refresh_annotations()
    
    def refresh_annotations(self):
        """Refresh annotations display"""
        self.clear()
        
        for i, ann in enumerate(self.annotation_manager.annotations):
            if ann.is_point_event():
                text = f"{ann.start_time:.3f}s - {ann.category}: {ann.description}"
            else:
                duration_ms = ann.duration * 1000
                text = f"{ann.start_time:.3f}s-{ann.end_time:.3f}s ({duration_ms:.1f}ms) - {ann.category}: {ann.description}"
            
             #Add channel name if present
            if ann.channel_name:
                text += f" [{ann.channel_name}]"
            self.addItem(text)
    
    def on_item_double_clicked(self, item):
        """Handle double-click to navigate to time"""
        row = self.row(item)
        if row < len(self.annotation_manager.annotations):
            # Navigate to the start time of the event
            time = self.annotation_manager.annotations[row].start_time
            self.annotation_clicked.emit(time)
    
    def show_context_menu(self, pos):
        """Show context menu for edit/delete"""
        item = self.itemAt(pos)
        if item:
            from PyQt5.QtWidgets import QMenu
            menu = QMenu(self)
            
            edit_action = menu.addAction("Edit")
            delete_action = menu.addAction("Delete")
            
            action = menu.exec_(self.mapToGlobal(pos))
            
            row = self.row(item)
            if action == edit_action:
                self.edit_annotation(row)
            elif action == delete_action:
                self.delete_annotation(row)
            elif action == navigate_action:
                if row < len(self.annotation_manager.annotations):
                    time = self.annotation_manager.annotations[row].start_time
                    self.annotation_clicked.emit(time)
    
    def edit_annotation(self, row):
        """Edit annotation at row"""
        if row < len(self.annotation_manager.annotations):
            annotation = self.annotation_manager.annotations[row]
            dialog = AnnotationDialog(
                annotation, 
                self.annotation_manager.categories, 
                self,
                current_time=annotation.start_time
            )
            
            if dialog.exec_() == QDialog.Accepted:
                data = dialog.get_annotation_data()
                if data:
                    # Update annotation
                    annotation.start_time = data['start_time']
                    annotation.end_time = data['end_time']
                    annotation.description = data['description']
                    annotation.category = data['category']
                    annotation.channel_name = data['channel_name']
                    self.annotation_manager.annotations_changed.emit()
    
    def delete_annotation(self, row):
        """Delete annotation at row"""
        if row < len(self.annotation_manager.annotations):
            annotation = self.annotation_manager.annotations[row]
            
            if annotation.is_point_event():
                msg = f"Delete event at {annotation.start_time:.3f}s?"
            else:
                msg = f"Delete event from {annotation.start_time:.3f}s to {annotation.end_time:.3f}s ({annotation.duration*1000:.1f}ms)?"
            
            reply = QMessageBox.question(
                self, "Delete Event", msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.annotation_manager.remove_annotation(row)

class AnnotationControls(QHBoxLayout):
    """Control buttons for annotations"""
    
    def __init__(self, annotation_manager, parent_widget):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.parent_widget = parent_widget
        
        # Add button
        self.add_btn = QPushButton("Add Event")
        self.add_btn.clicked.connect(self.add_annotation)
        self.addWidget(self.add_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_annotations)
        self.addWidget(self.clear_btn)
        
        # Export button
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_annotations)
        self.addWidget(self.export_btn)
        
        # Statistics label
        self.stats_label = QLabel("0 events")
        self.addWidget(self.stats_label)
        
        # Update stats when annotations change
        self.annotation_manager.annotations_changed.connect(self.update_stats)
        self.update_stats()
    
    def add_annotation(self):
        """Add new annotation at current time"""
        # Get current time from parent
        current_time = getattr(self.parent_widget, 'current_time', 0.0)
        
        dialog = AnnotationDialog(
            categories=self.annotation_manager.categories, 
            parent=self.parent_widget,
            current_time=current_time
        )
        
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_annotation_data()
            if data:
                self.annotation_manager.add_annotation(
                    start_time=data['start_time'],
                    end_time=data['end_time'],
                    description=data['description'],
                    category=data['category'],
                    channel_name=data['channel_name']
                )
    
    def clear_annotations(self):
        """Clear all annotations"""
        if self.annotation_manager.annotations:
            reply = QMessageBox.question(
                self.parent_widget, "Clear Events",
                f"Delete all {len(self.annotation_manager.annotations)} events?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.annotation_manager.clear_annotations()
    
    def export_annotations(self):
        """Export annotations to CSV - defaults to original data folder"""
        if not self.annotation_manager.annotations:
            QMessageBox.warning(self.parent_widget, "No Events", "No events to export.")
            return
        
        # Get default save path from original data location
        default_path = "events.csv"
        try:
            if (hasattr(self.parent_widget, 'datasets') and 
                hasattr(self.parent_widget, 'current_index') and
                self.parent_widget.current_index >= 0):
                
                original_file = self.parent_widget.datasets[self.parent_widget.current_index].get("file_path", "")
                if original_file and isinstance(original_file, str):
                    import os
                    # Get directory of original file
                    original_dir = os.path.dirname(original_file)
                    # Get base name without extension
                    base_name = os.path.splitext(os.path.basename(original_file))[0]
                    # Create default filename: original_name_events.csv
                    default_path = os.path.join(original_dir, f"{base_name}_events.csv")
        except Exception as e:
            print(f"Could not get original path: {e}")
            default_path = "events.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent_widget, "Export Events", default_path, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.annotation_manager.export_to_csv(file_path)
                QMessageBox.information(
                    self.parent_widget, "Export Successful", 
                    f"Exported {len(self.annotation_manager.annotations)} events to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self.parent_widget, "Export Error", f"Failed to export:\n{str(e)}")
        
    def update_stats(self):
        """Update statistics display"""
        total = len(self.annotation_manager.annotations)
        ripples = len(self.annotation_manager.get_annotations_by_category('ripple'))
        spindles = len(self.annotation_manager.get_annotations_by_category('spindle'))
        deltas = len(self.annotation_manager.get_annotations_by_category('delta'))
        
        # Calculate total duration for duration events
        total_duration = sum(ann.duration for ann in self.annotation_manager.annotations 
                           if not ann.is_point_event())
        
        if total_duration > 0:
            self.stats_label.setText(
                f"{total} events (R:{ripples} S:{spindles} D:{deltas}) | "
                f"Total: {total_duration:.2f}s"
            )
        else:
            self.stats_label.setText(f"{total} events (R:{ripples} S:{spindles} D:{deltas})")


# Drag-to-mark annotation class
class DragToMarkAnnotation(QObject):
    """Handle drag-to-mark annotation on PyQtGraph plot widget"""
    
    def __init__(self, plot_widget, annotation_manager):
        super().__init__()
        self.plot_widget = plot_widget
        self.annotation_manager = annotation_manager
        
        # State tracking
        self.is_dragging = False
        self.drag_start_time = None
        self.drag_region = None
        self.drag_complete = False 
        self.drag_threshold = 0.005  # 5ms threshold to consider as point event
        
        # Store permanent annotation regions
        self.annotation_regions = []

        # Connect mouse events
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        
        # Connect to annotation changes to update visual regions
        self.annotation_manager.annotations_changed.connect(self.update_annotation_regions)
        
        print("Drag-to-mark annotation enabled:")
        print("  â€¢ Left-click and drag to select time region")
        print("  â€¢ Release to open annotation dialog")
    
    def on_mouse_click(self, event):
        """Handle mouse click events"""
        # Check if click is on the plot area
        if self.plot_widget.sceneBoundingRect().contains(event.scenePos()):
            pos = event.scenePos()
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            time = mouse_point.x()
            
            if event.button() == 1:  # Left click - drag to mark
                if not self.is_dragging:
                    # Start dragging
                    self.start_drag(time)
                else:
                    # End dragging
                    self.end_drag(time)
    
    def on_mouse_move(self, pos):
        """Handle mouse move during drag"""
        if self.is_dragging and self.drag_region is not None:
            # Update drag region
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            current_time = mouse_point.x()
            
            # Update the region
            start = min(self.drag_start_time, current_time)
            end = max(self.drag_start_time, current_time)
            self.drag_region.setRegion([start, end])
    
    def start_drag(self, time):
        """Start dragging to select region"""
        self.is_dragging = True
        self.drag_start_time = time
        
        # Create visual feedback region with light grey color
        self.drag_region = pg.LinearRegionItem(
            [time, time],
            brush=(200, 200, 200, 100),  # Light grey
            pen=pg.mkPen((150, 150, 150), width=2),  # Darker grey border
            movable=False
        )
        self.plot_widget.addItem(self.drag_region)
    
    def end_drag(self, end_time):
        """End dragging and open annotation dialog"""
        if not self.is_dragging or self.drag_region is None:
            return
        
        self.is_dragging = False
        
        # Get start and end times
        start_time = min(self.drag_start_time, end_time)
        end_time = max(self.drag_start_time, end_time)
        
        # Remove visual feedback
        self.plot_widget.removeItem(self.drag_region)
        self.drag_region = None
        
        # Check if it's effectively a point (very small drag)
        if abs(end_time - start_time) < 0.005:  # Less than 5ms
            # Treat as point event
            end_time = start_time
        
        # Open annotation dialog with pre-filled times
        self.open_annotation_dialog(start_time, end_time)
    
    def open_annotation_dialog(self, start_time, end_time):
        """Open annotation dialog with pre-filled times"""
        dialog = AnnotationDialog(
            annotation=None,
            categories=self.annotation_manager.categories,
            parent=self.plot_widget,
            current_time=start_time
        )
        
        # Pre-fill the times
        dialog.start_time_edit.setValue(start_time)
        dialog.end_time_edit.setValue(end_time)
        
        # Focus on description field for quick input
        dialog.desc_edit.setFocus()
        
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_annotation_data()
            if data:
                self.annotation_manager.add_annotation(
                    start_time=data['start_time'],
                    end_time=data['end_time'],
                    description=data['description'],
                    category=data['category'],
                    channel_name=data['channel_name']
                )
    
    def update_annotation_regions(self):
        """Update visual regions for all annotations"""
        # Clear existing regions
        for region in self.annotation_regions:
            self.plot_widget.removeItem(region)
        self.annotation_regions.clear()
        
        # Create regions for all annotations
        for ann in self.annotation_manager.annotations:
            if not ann.is_point_event():  # Only show regions for duration events
                region = pg.LinearRegionItem(
                    [ann.start_time, ann.end_time],
                    brush=(180, 180, 180, 80),  # Gray shadow
                    pen=None,  # No border for cleaner look
                    movable=False
                )
                region.setZValue(-10)  # Behind the data curves
                self.plot_widget.addItem(region)
                self.annotation_regions.append(region)


def add_drag_to_mark_annotation(plot_widget, annotation_manager):
    """
    Add drag-to-mark annotation capability to PyQtGraph plot widget
    
    Usage:
        - Left-click and drag on plot to select time region
        - Release to open annotation dialog with times pre-filled
    
    Returns:
        DragToMarkAnnotation instance
    """
    return DragToMarkAnnotation(plot_widget, annotation_manager)


# Keep old function for backward compatibility but redirect to new system
def add_right_click_annotation(plot_widget, annotation_manager):
    """Legacy function - redirects to drag-to-mark system"""
    print("Note: Using new left-click drag-to-mark annotation system")
    return add_drag_to_mark_annotation(plot_widget, annotation_manager)


# Test function
if __name__ == "__main__":
    print("Annotation module loaded successfully!")
    print("Available classes: AnnotationManager, AnnotationWidget, AnnotationDialog")
    
    # Simple test
    manager = AnnotationManager()
    manager.add_annotation(1.5, 1.565, "Test ripple", "ripple")
    manager.add_annotation(3.2, 3.35, "Test spindle", "spindle")
    manager.add_annotation(5.0, 5.0, "Point event", "other")  # Point event
    
    print(f"\nTest: Added {len(manager.annotations)} annotations")
    for ann in manager.annotations:
        if ann.is_point_event():
            print(f"  {ann.start_time}s - {ann.category}: {ann.description} [Point event]")
        else:
            print(f"  {ann.start_time}s-{ann.end_time}s ({ann.duration*1000:.1f}ms) - {ann.category}: {ann.description}")