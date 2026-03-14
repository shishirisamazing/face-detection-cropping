# -*- coding: utf-8 -*-
#author: Tomo Lapautre

# Import facecrop before PyQt5 — PyQt5 interferes with onnxruntime provider
# loading, which breaks rembg's background removal detection.
from main.facecrop import FaceCrop
from main.constants import CV2_FILETYPES

import glob
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageOps
from itertools import compress
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


param_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.json') #file used to save cropping preferences

try:
    with open(param_file_path) as parameters_json:
        parameters = json.load(parameters_json)
    autorisation = True
except Exception:
    autorisation = False
    print("No autorisation to read parameters.json")


DARK_STYLESHEET = """
/* ===== Global ===== */
* {
    font-family: "Segoe UI", sans-serif;
}

QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: transparent;
    color: #e0e0e0;
}

/* ===== Sidebar ===== */
QWidget#sidebar {
    background-color: #2b2b2b;
    border-right: 1px solid #3c3c3c;
}

/* ===== Section Headers ===== */
QLabel#section_header {
    color: #999999;
    font-size: 8pt;
    font-weight: bold;
    text-transform: uppercase;
    padding: 0px;
    letter-spacing: 1px;
}

/* ===== Preview Panel ===== */
QWidget#preview_panel {
    background-color: #1e1e1e;
}

QLabel#preview_image {
    background-color: #141414;
    border: 1px solid #333333;
    border-radius: 4px;
}

QLabel#title {
    color: #ffffff;
    font-size: 14pt;
    font-weight: bold;
    padding: 0px;
}

QLabel#path_label {
    color: #777777;
    font-size: 8pt;
}

/* ===== Line Edits ===== */
QLineEdit {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #444444;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 10pt;
    selection-background-color: #0078d4;
}

QLineEdit:focus {
    border: 1px solid #0078d4;
}

/* ===== Combo Boxes ===== */
QComboBox {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #444444;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 10pt;
}

QComboBox:focus {
    border: 1px solid #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #444444;
    selection-background-color: #0078d4;
}

/* ===== Buttons ===== */
QPushButton {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 9pt;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #444444;
    border: 1px solid #555555;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton#crop_button {
    background-color: #0078d4;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 11pt;
    font-weight: bold;
}

QPushButton#crop_button:hover {
    background-color: #1a8aff;
}

QPushButton#crop_button:pressed {
    background-color: #005fa3;
}

QPushButton#preview_button {
    background-color: transparent;
    border: 1px solid #0078d4;
    color: #0078d4;
    font-weight: bold;
    padding: 8px 20px;
}

QPushButton#preview_button:hover {
    background-color: #0078d4;
    color: #ffffff;
}

/* ===== Checkboxes ===== */
QCheckBox {
    color: #cccccc;
    spacing: 6px;
    font-size: 9pt;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 2px;
    border: 1px solid #555555;
    background-color: #1e1e1e;
}

QCheckBox::indicator:checked {
    background-color: #0078d4;
    border: 1px solid #0078d4;
}

QCheckBox::indicator:hover {
    border: 1px solid #0078d4;
}

/* ===== Color Picker Button ===== */
QPushButton#bg_color_button {
    min-width: 28px;
    max-width: 28px;
    min-height: 20px;
    max-height: 20px;
    border: 2px solid #555555;
    border-radius: 3px;
    padding: 0px;
}

QPushButton#bg_color_button:hover {
    border: 2px solid #0078d4;
}

/* ===== Thumbnails ===== */
QFrame#thumbnail {
    background-color: #2a2a2a;
    border: 2px solid #3a3a3a;
    border-radius: 6px;
}

QFrame#thumbnail:hover {
    border: 2px solid #0078d4;
    background-color: #303030;
}

/* ===== Back Button ===== */
QPushButton#back_button {
    background-color: transparent;
    border: none;
    color: #0078d4;
    font-size: 9pt;
    font-weight: bold;
    padding: 4px 8px;
    text-align: left;
}

QPushButton#back_button:hover {
    color: #1a8aff;
}

/* ===== Progress Dialog ===== */
QProgressDialog {
    background-color: #2b2b2b;
    color: #e0e0e0;
}

QProgressBar {
    background-color: #1e1e1e;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    text-align: center;
    color: #e0e0e0;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 2px;
}

/* ===== Message Box ===== */
QMessageBox {
    background-color: #2b2b2b;
    color: #e0e0e0;
}

QMessageBox QLabel {
    color: #e0e0e0;
}

QMessageBox QPushButton {
    min-width: 80px;
}

/* ===== Separator ===== */
QFrame#separator {
    background-color: #3c3c3c;
    max-height: 1px;
}

/* ===== Scroll Bars ===== */
QScrollArea {
    border: none;
}

QScrollBar:vertical {
    background: #2b2b2b;
    width: 8px;
}

QScrollBar::handle:vertical {
    background: #555555;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #666666;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ===== Status Bar ===== */
QStatusBar {
    background-color: #1e1e1e;
    color: #666666;
    border-top: 1px solid #333333;
    font-size: 8pt;
}

/* ===== Menu Bar ===== */
QMenuBar {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border-bottom: 1px solid #333333;
}
"""


THUMB_SIZE = 120


class ThumbnailLoaderThread(QtCore.QThread):
    """Loads image thumbnails in a background thread."""
    thumbnail_ready = QtCore.pyqtSignal(str, QtGui.QImage)

    def __init__(self, file_paths, thumb_size=THUMB_SIZE, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.thumb_size = thumb_size
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        for path in self.file_paths:
            if self._abort:
                break
            try:
                pil_img = Image.open(path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img.thumbnail((self.thumb_size, self.thumb_size), Image.LANCZOS)
                if pil_img.mode == 'RGBA':
                    data = pil_img.tobytes('raw', 'RGBA')
                    qimg = QtGui.QImage(data, pil_img.width, pil_img.height,
                                        4 * pil_img.width, QtGui.QImage.Format_RGBA8888)
                else:
                    pil_img = pil_img.convert('RGB')
                    data = pil_img.tobytes('raw', 'RGB')
                    qimg = QtGui.QImage(data, pil_img.width, pil_img.height,
                                        3 * pil_img.width, QtGui.QImage.Format_RGB888)
                qimg = qimg.copy()  # deep copy so data outlives this iteration
                self.thumbnail_ready.emit(path, qimg)
            except Exception:
                pass


class SpinnerOverlay(QtWidgets.QWidget):
    """Semi-transparent overlay with an animated spinning circle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.setInterval(30)
        self._message = "Processing..."
        self.hide()
        if parent:
            parent.installEventFilter(self)

    def start(self, message="Processing..."):
        self._message = message
        self._angle = 0
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() == QtCore.QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Semi-transparent dark background
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 140))

        # Draw spinning arc
        center = self.rect().center()
        radius = 24
        pen = QtGui.QPen(QtGui.QColor(0, 120, 212), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        arc_rect = QtCore.QRectF(center.x() - radius, center.y() - radius - 14,
                                  radius * 2, radius * 2)
        painter.drawArc(arc_rect, self._angle * 16, 270 * 16)

        # Draw message text
        painter.setPen(QtGui.QColor(224, 224, 224))
        font = QtGui.QFont("Segoe UI", 10)
        painter.setFont(font)
        text_rect = QtCore.QRectF(0, center.y() + radius, self.width(), 30)
        painter.drawText(text_rect, Qt.AlignCenter, self._message)

        painter.end()


class PreviewWorkerThread(QtCore.QThread):
    """Runs face detection + crop in a background thread for preview."""
    finished = QtCore.pyqtSignal(object, str, object)  # (PIL Image or None, file_path, FaceCrop)
    error = QtCore.pyqtSignal(str, str)                # (error_type, error_message)

    def __init__(self, file_path, kwargs, bg_color, cached_facecrop=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.kwargs = kwargs
        self.bg_color = bg_color
        self._cached_facecrop = cached_facecrop

    def run(self):
        face_crop = self._cached_facecrop
        if face_crop is None:
            try:
                face_crop = FaceCrop(bg_color=self.bg_color, **self.kwargs)
            except Exception as e:
                self.error.emit("init", str(e))
                return

        try:
            result = face_crop.crop_single(self.file_path)
        except Exception as e:
            self.error.emit("crop", str(e))
            return

        self.finished.emit(result, self.file_path, face_crop)


class CropInitWorkerThread(QtCore.QThread):
    """Runs FaceCrop initialization in a background thread."""
    finished = QtCore.pyqtSignal(object)    # FaceCrop instance
    error = QtCore.pyqtSignal(str)

    def __init__(self, kwargs, bg_color, pyqt_ui, cached_facecrop=None, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs
        self.bg_color = bg_color
        self.pyqt_ui = pyqt_ui
        self._cached_facecrop = cached_facecrop

    def run(self):
        if self._cached_facecrop is not None:
            # Reuse cached instance, just update pyqt_ui reference
            self._cached_facecrop.pyqt_ui = self.pyqt_ui
            self._cached_facecrop.progress_count = 0
            self.finished.emit(self._cached_facecrop)
            return
        try:
            face_crop = FaceCrop(pyqt_ui=self.pyqt_ui, bg_color=self.bg_color, **self.kwargs)
            self.finished.emit(face_crop)
        except Exception as e:
            self.error.emit(str(e))


class ClickableThumbnail(QtWidgets.QFrame):
    """A clickable thumbnail tile showing an image and its filename."""
    clicked = QtCore.pyqtSignal(str)
    removed = QtCore.pyqtSignal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._selected = False
        self.setObjectName("thumbnail")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.setFixedSize(THUMB_SIZE + 16, THUMB_SIZE + 34)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.img_label.setStyleSheet("background-color: #1a1a1a; border-radius: 2px; border: none;")
        self.img_label.setText("...")
        layout.addWidget(self.img_label, 0, Qt.AlignCenter)

        name_label = QtWidgets.QLabel()
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("color: #888; font-size: 7pt; background: transparent; border: none;")
        metrics = QtGui.QFontMetrics(name_label.font())
        elided = metrics.elidedText(Path(file_path).stem, Qt.ElideMiddle, THUMB_SIZE)
        name_label.setText(elided)
        name_label.setToolTip(Path(file_path).name)
        layout.addWidget(name_label, 0, Qt.AlignCenter)

        # Remove button — shown on hover
        self._remove_btn = QtWidgets.QPushButton("\u00d7", self)
        self._remove_btn.setFixedSize(20, 20)
        self._remove_btn.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self._remove_btn.setStyleSheet(
            "QPushButton { background-color: rgba(0,0,0,180); color: #fff; border: none;"
            " border-radius: 10px; font-size: 12pt; font-weight: bold; padding: 0px; }"
            "QPushButton:hover { background-color: #e04040; }"
        )
        self._remove_btn.move(self.width() - 24, 4)
        self._remove_btn.hide()
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self.file_path))

    def set_pixmap(self, pixmap):
        self.img_label.setText("")
        self.img_label.setPixmap(pixmap.scaled(
            THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def set_selected(self, selected):
        self._selected = selected
        if selected:
            self.setStyleSheet(
                "QFrame#thumbnail { border: 2px solid #0078d4; background-color: #1a3050; }"
            )
        else:
            self.setStyleSheet("")

    def enterEvent(self, event):
        self._remove_btn.show()
        self._remove_btn.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._remove_btn.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self.file_path)
        super().mousePressEvent(event)


class Ui_MainWindow(object):
    def __init__(self, language):
        self.language = language
        self._selected_file = None
        self._thumbnail_widgets = {}  # file_path -> ClickableThumbnail
        self._thumb_loader_thread = None
        self._preview_worker = None
        self._crop_init_worker = None
        self._excluded_files = set()  # files removed from gallery by user
        self._cached_facecrop = None      # cached FaceCrop instance
        self._cached_facecrop_key = None  # (kwargs_tuple, bg_color) used to create it

    def _make_separator(self):
        sep = QtWidgets.QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFixedHeight(1)
        return sep

    def _make_section_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("section_header")
        return label

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1050, 700)
        MainWindow.setMinimumSize(QtCore.QSize(800, 500))

        central = QtWidgets.QWidget(MainWindow)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ==========================================
        # LEFT SIDEBAR — all controls
        # ==========================================
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(300)

        # Scroll area for sidebar content
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_content = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(sidebar_content)
        side.setSpacing(8)
        side.setContentsMargins(14, 14, 14, 14)

        # Title
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("title")
        side.addWidget(self.title)

        side.addSpacing(4)

        # --- FOLDERS ---
        side.addWidget(self._make_section_label("FOLDERS"))
        side.addSpacing(2)

        self.input_dir_button = QtWidgets.QPushButton(clicked=lambda: self.select_input_dir())
        self.input_dir_button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        side.addWidget(self.input_dir_button)

        self.input_path_label = QtWidgets.QLabel("No folder selected")
        self.input_path_label.setObjectName("path_label")
        self.input_path_label.setWordWrap(True)
        side.addWidget(self.input_path_label)

        side.addSpacing(2)

        self.output_dir_button = QtWidgets.QPushButton(clicked=lambda: self.select_output_dir())
        self.output_dir_button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        side.addWidget(self.output_dir_button)

        self.output_path_label = QtWidgets.QLabel("No folder selected")
        self.output_path_label.setObjectName("path_label")
        self.output_path_label.setWordWrap(True)
        side.addWidget(self.output_path_label)

        side.addWidget(self._make_separator())

        # --- CROP SETTINGS ---
        side.addWidget(self._make_section_label("CROP SETTINGS"))
        side.addSpacing(2)

        font_label = QtGui.QFont("Segoe UI", 9)
        font_input = QtGui.QFont("Segoe UI", 9)
        font_input.setBold(True)

        # Mode selector
        mode_form = QtWidgets.QFormLayout()
        mode_form.setHorizontalSpacing(10)
        mode_form.setVerticalSpacing(6)
        mode_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.crop_mode_label = QtWidgets.QLabel()
        self.crop_mode_label.setFont(font_label)
        self.crop_mode_combo = QtWidgets.QComboBox()
        self.crop_mode_combo.setFont(font_input)
        self.crop_mode_combo.addItems(["Percentage", "Aspect Ratio", "Custom Pixels"])
        self.crop_mode_combo.currentIndexChanged.connect(self._on_crop_mode_changed)
        mode_form.addRow(self.crop_mode_label, self.crop_mode_combo)
        side.addLayout(mode_form)

        # Stacked widget for mode-specific fields
        self.crop_mode_stack = QtWidgets.QStackedWidget()

        # Page 0: Percentage
        pct_page = QtWidgets.QWidget()
        pct_form = QtWidgets.QFormLayout(pct_page)
        pct_form.setHorizontalSpacing(10)
        pct_form.setVerticalSpacing(6)
        pct_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct_form.setContentsMargins(0, 4, 0, 4)

        self.width_label = QtWidgets.QLabel()
        self.width_label.setFont(font_label)
        self.width_input = QtWidgets.QLineEdit('60')
        self.width_input.setFont(font_input)
        pct_form.addRow(self.width_label, self.width_input)

        self.height_label = QtWidgets.QLabel()
        self.height_label.setFont(font_label)
        self.height_input = QtWidgets.QLineEdit('60')
        self.height_input.setFont(font_input)
        pct_form.addRow(self.height_label, self.height_input)

        self.crop_mode_stack.addWidget(pct_page)  # index 0

        # Page 1: Aspect Ratio
        ratio_page = QtWidgets.QWidget()
        ratio_form = QtWidgets.QFormLayout(ratio_page)
        ratio_form.setHorizontalSpacing(10)
        ratio_form.setVerticalSpacing(6)
        ratio_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ratio_form.setContentsMargins(0, 4, 0, 4)

        self.ratio_label = QtWidgets.QLabel()
        self.ratio_label.setFont(font_label)
        self.ratio_combo = QtWidgets.QComboBox()
        self.ratio_combo.setFont(font_input)
        self.ratio_combo.addItems(["1:1", "4:3", "3:4", "16:9", "9:16", "Custom"])
        self.ratio_combo.currentTextChanged.connect(self._on_ratio_preset_changed)
        ratio_form.addRow(self.ratio_label, self.ratio_combo)

        self.ratio_custom_w_label = QtWidgets.QLabel()
        self.ratio_custom_w_label.setFont(font_label)
        self.ratio_custom_w_input = QtWidgets.QLineEdit('3')
        self.ratio_custom_w_input.setFont(font_input)
        self.ratio_custom_w_label.hide()
        self.ratio_custom_w_input.hide()
        ratio_form.addRow(self.ratio_custom_w_label, self.ratio_custom_w_input)

        self.ratio_custom_h_label = QtWidgets.QLabel()
        self.ratio_custom_h_label.setFont(font_label)
        self.ratio_custom_h_input = QtWidgets.QLineEdit('4')
        self.ratio_custom_h_input.setFont(font_input)
        self.ratio_custom_h_label.hide()
        self.ratio_custom_h_input.hide()
        ratio_form.addRow(self.ratio_custom_h_label, self.ratio_custom_h_input)

        self.padding_label = QtWidgets.QLabel()
        self.padding_label.setFont(font_label)
        self.padding_input = QtWidgets.QLineEdit('2.5')
        self.padding_input.setFont(font_input)
        ratio_form.addRow(self.padding_label, self.padding_input)

        self.crop_mode_stack.addWidget(ratio_page)  # index 1

        # Page 2: Custom Pixels
        px_page = QtWidgets.QWidget()
        px_form = QtWidgets.QFormLayout(px_page)
        px_form.setHorizontalSpacing(10)
        px_form.setVerticalSpacing(6)
        px_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        px_form.setContentsMargins(0, 4, 0, 4)

        self.px_width_label = QtWidgets.QLabel()
        self.px_width_label.setFont(font_label)
        self.px_width_input = QtWidgets.QLineEdit('600')
        self.px_width_input.setFont(font_input)
        px_form.addRow(self.px_width_label, self.px_width_input)

        self.px_height_label = QtWidgets.QLabel()
        self.px_height_label.setFont(font_label)
        self.px_height_input = QtWidgets.QLineEdit('800')
        self.px_height_input.setFont(font_input)
        px_form.addRow(self.px_height_label, self.px_height_input)

        self.crop_mode_stack.addWidget(px_page)  # index 2

        side.addWidget(self.crop_mode_stack)

        # Common fields (all modes)
        common_form = QtWidgets.QFormLayout()
        common_form.setHorizontalSpacing(10)
        common_form.setVerticalSpacing(6)
        common_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.width_asy_label = QtWidgets.QLabel()
        self.width_asy_label.setFont(font_label)
        self.width_asy_input = QtWidgets.QLineEdit('0')
        self.width_asy_input.setFont(font_input)
        common_form.addRow(self.width_asy_label, self.width_asy_input)

        self.height_asy_label = QtWidgets.QLabel()
        self.height_asy_label.setFont(font_label)
        self.height_asy_input = QtWidgets.QLineEdit('0')
        self.height_asy_input.setFont(font_input)
        common_form.addRow(self.height_asy_label, self.height_asy_input)

        self.tag_label = QtWidgets.QLabel()
        self.tag_label.setFont(font_label)
        self.tag_input = QtWidgets.QLineEdit()
        self.tag_input.setFont(font_input)
        common_form.addRow(self.tag_label, self.tag_input)

        side.addLayout(common_form)

        side.addWidget(self._make_separator())

        # --- OPTIONS ---
        side.addWidget(self._make_section_label("OPTIONS"))
        side.addSpacing(2)

        self.checkbox_count = QtWidgets.QCheckBox()
        self.checkbox_count.setChecked(True)
        self.checkbox_count.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        side.addWidget(self.checkbox_count)

        self.checkbox_folder = QtWidgets.QCheckBox()
        self.checkbox_folder.setChecked(False)
        self.checkbox_folder.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        side.addWidget(self.checkbox_folder)

        side.addSpacing(4)

        # Background replacement
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.setSpacing(6)
        self.checkbox_bg = QtWidgets.QCheckBox()
        self.checkbox_bg.setChecked(False)
        self.checkbox_bg.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.checkbox_bg.toggled.connect(self._on_bg_toggled)
        bg_row.addWidget(self.checkbox_bg)

        self.bg_color_button = QtWidgets.QPushButton()
        self.bg_color_button.setObjectName("bg_color_button")
        self.bg_color_button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.bg_color_button.clicked.connect(self._pick_bg_color)
        self.bg_color_button.setEnabled(False)
        self._selected_bg_color = QtGui.QColor(255, 255, 255)
        self._update_color_button()
        bg_row.addWidget(self.bg_color_button)
        bg_row.addStretch()
        side.addLayout(bg_row)

        side.addWidget(self._make_separator())

        # --- ACTION BUTTONS ---
        side.addSpacing(4)

        self.preview_button = QtWidgets.QPushButton(clicked=lambda: self.preview_selected())
        self.preview_button.setObjectName("preview_button")
        self.preview_button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.preview_button.setMinimumHeight(34)
        side.addWidget(self.preview_button)

        side.addSpacing(4)

        self.crop_button = QtWidgets.QPushButton(clicked=lambda: self.crop_all())
        self.crop_button.setObjectName("crop_button")
        self.crop_button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.crop_button.setMinimumHeight(40)
        side.addWidget(self.crop_button)

        side.addStretch()

        sidebar_scroll.setWidget(sidebar_content)
        sidebar_outer = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_outer.setContentsMargins(0, 0, 0, 0)
        sidebar_outer.addWidget(sidebar_scroll)

        root_layout.addWidget(self.sidebar)

        # ==========================================
        # RIGHT PANEL — gallery + preview
        # ==========================================
        preview_panel = QtWidgets.QWidget()
        preview_panel.setObjectName("preview_panel")
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(16, 16, 16, 16)

        self.right_stack = QtWidgets.QStackedWidget()

        # --- Page 0: Gallery ---
        self.gallery_page = QtWidgets.QWidget()
        gallery_layout = QtWidgets.QVBoxLayout(self.gallery_page)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(8)

        # Image count label
        self.gallery_count_label = QtWidgets.QLabel()
        self.gallery_count_label.setObjectName("section_header")
        gallery_layout.addWidget(self.gallery_count_label)

        self.gallery_scroll = QtWidgets.QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.gallery_container = QtWidgets.QWidget()
        self.gallery_grid = QtWidgets.QGridLayout(self.gallery_container)
        self.gallery_grid.setSpacing(8)
        self.gallery_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.gallery_scroll.setWidget(self.gallery_container)

        # Placeholder text
        self.gallery_placeholder = QtWidgets.QLabel("Select an input folder to browse images")
        self.gallery_placeholder.setAlignment(Qt.AlignCenter)
        self.gallery_placeholder.setStyleSheet("color: #555; font-size: 11pt;")
        self.gallery_placeholder.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        gallery_layout.addWidget(self.gallery_placeholder)
        gallery_layout.addWidget(self.gallery_scroll)
        self.gallery_scroll.hide()
        self.gallery_count_label.hide()

        self.right_stack.addWidget(self.gallery_page)  # index 0

        # --- Page 1: Preview ---
        self.preview_page = QtWidgets.QWidget()
        pv_layout = QtWidgets.QVBoxLayout(self.preview_page)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        pv_layout.setSpacing(8)

        back_btn = QtWidgets.QPushButton("\u2190  Back to Images")
        back_btn.setObjectName("back_button")
        back_btn.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        back_btn.clicked.connect(lambda: self.right_stack.setCurrentIndex(0))
        back_btn.setFixedHeight(30)
        pv_layout.addWidget(back_btn, 0, Qt.AlignLeft)

        self.preview_image = QtWidgets.QLabel()
        self.preview_image.setObjectName("preview_image")
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.preview_image.setMinimumSize(QtCore.QSize(300, 300))
        pv_layout.addWidget(self.preview_image)

        self.preview_filename = QtWidgets.QLabel()
        self.preview_filename.setAlignment(Qt.AlignCenter)
        self.preview_filename.setStyleSheet("color: #888; font-size: 9pt;")
        pv_layout.addWidget(self.preview_filename)

        self.right_stack.addWidget(self.preview_page)  # index 1

        preview_layout.addWidget(self.right_stack)
        root_layout.addWidget(preview_panel, 1)

        # Spinner overlay for loading states (child of right panel)
        self._spinner = SpinnerOverlay(preview_panel)

        # ==========================================
        # Finalize
        # ==========================================
        MainWindow.setCentralWidget(central)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        #loads parameters if app managed to load parameters.json file
        if autorisation:
            self.tag_input.setText(parameters['tag'])
            self.height_asy_input.setText(str(parameters['height_asy']))
            self.width_asy_input.setText(str(parameters['width_asy']))
            self.height_input.setText(str(parameters["height"]))
            self.width_input.setText(str(parameters['width']))
            self.checkbox_folder.setChecked(parameters['folder_option'])
            self.checkbox_count.setChecked(parameters['single_face_option'])

            crop_mode = parameters.get('crop_mode', 0)
            self.crop_mode_combo.setCurrentIndex(crop_mode)
            self.crop_mode_stack.setCurrentIndex(crop_mode)

            ratio_preset = parameters.get('ratio_preset', '1:1')
            idx = self.ratio_combo.findText(ratio_preset)
            if idx >= 0:
                self.ratio_combo.setCurrentIndex(idx)

            self.ratio_custom_w_input.setText(str(parameters.get('ratio_custom_w', '3')))
            self.ratio_custom_h_input.setText(str(parameters.get('ratio_custom_h', '4')))
            self.padding_input.setText(str(parameters.get('padding_multiplier', '2.5')))
            self.px_width_input.setText(str(parameters.get('custom_width_px', '600')))
            self.px_height_input.setText(str(parameters.get('custom_height_px', '800')))

        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _t = QtCore.QCoreApplication.translate

        if self.language == "french":
            MainWindow.setWindowTitle("FaceCrop")
            self.title.setText("Cadrage Automatique")
            self.input_dir_button.setText("Dossier des Images")
            self.output_dir_button.setText("Dossier de Sortie")
            self.crop_mode_label.setText("Mode")
            self.width_label.setText("Largeur (%)")
            self.height_label.setText("Hauteur (%)")
            self.ratio_label.setText("Ratio")
            self.ratio_custom_w_label.setText("Ratio L")
            self.ratio_custom_h_label.setText("Ratio H")
            self.padding_label.setText("Marge")
            self.px_width_label.setText("Largeur (px)")
            self.px_height_label.setText("Hauteur (px)")
            self.width_asy_label.setText("Assym\u00e9trie H (%)")
            self.height_asy_label.setText("Assym\u00e9trie V (%)")
            self.tag_label.setText("Tag Fichier")
            self.checkbox_count.setText("Un visage par photo")
            self.checkbox_folder.setText("Un dossier par visage")
            self.checkbox_bg.setText("Arri\u00e8re-plan")
            self.preview_button.setText("Aper\u00e7u")
            self.crop_button.setText("Rogner Tout")
            self.input_path_label.setText("Aucun dossier")
            self.output_path_label.setText("Aucun dossier")
            self.warning_values = 'Valeurs non reconnues'
            self.warning_init = "\u00c9chec de l'initialisation"
            self.warning_folders = "Veuillez indiquer un dossier d'entr\u00e9e et de sortie"
            self.warning_title = 'Erreur'
            self.warning_no_file = 'Votre dossier est vide'
            self.warning_no_face = 'Aucun visage d\u00e9tect\u00e9'
            self.gallery_placeholder.setText("S\u00e9lectionnez un dossier pour parcourir les images")
            self._text_processing_preview = "Traitement de l'aper\u00e7u..."
            self._text_loading_model = "Initialisation de la d\u00e9tection..."

        if self.language == "english":
            MainWindow.setWindowTitle("FaceCrop")
            self.title.setText("FaceCrop")
            self.input_dir_button.setText("Input Folder")
            self.output_dir_button.setText("Output Folder")
            self.crop_mode_label.setText("Mode")
            self.width_label.setText("Width (%)")
            self.height_label.setText("Height (%)")
            self.ratio_label.setText("Ratio")
            self.ratio_custom_w_label.setText("Ratio W")
            self.ratio_custom_h_label.setText("Ratio H")
            self.padding_label.setText("Padding")
            self.px_width_label.setText("Width (px)")
            self.px_height_label.setText("Height (px)")
            self.width_asy_label.setText("H. Asymmetry (%)")
            self.height_asy_label.setText("V. Asymmetry (%)")
            self.tag_label.setText("File Tag")
            self.checkbox_count.setText("One face per image")
            self.checkbox_folder.setText("Folder per face")
            self.checkbox_bg.setText("Replace background")
            self.preview_button.setText("Preview")
            self.crop_button.setText("Crop All")
            self.input_path_label.setText("No folder selected")
            self.output_path_label.setText("No folder selected")
            self.warning_folders = 'Please specify an input and output folder'
            self.warning_values = 'Input values not recognised'
            self.warning_init = 'Failed to initialize face detection'
            self.warning_title = 'Error'
            self.warning_no_file = 'Your input folder is empty'
            self.warning_no_face = 'No face detected in this image'
            self.gallery_placeholder.setText("Select an input folder to browse images")
            self._text_processing_preview = "Processing preview..."
            self._text_loading_model = "Initializing face detection..."


    def select_input_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory()
        if path:
            self.input_path = path
            self.paths = glob.glob('{}/*'.format(self.input_path))
            # show truncated path
            display = path if len(path) < 40 else "..." + path[-37:]
            self.input_path_label.setText(display)
            self.input_path_label.setToolTip(path)
            if len(self.paths) == 0:
                self.error_popup(self.warning_no_file)
            else:
                self.load_gallery()

    def select_output_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory()
        if path:
            self.output_path = path
            display = path if len(path) < 40 else "..." + path[-37:]
            self.output_path_label.setText(display)
            self.output_path_label.setToolTip(path)

    def load_gallery(self):
        """Load image thumbnails from the input folder into the gallery grid."""
        # Abort any running loader thread
        if self._thumb_loader_thread is not None:
            self._thumb_loader_thread.abort()
            self._thumb_loader_thread.wait()
            self._thumb_loader_thread = None

        # Clear existing thumbnails
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbnail_widgets.clear()
        self._selected_file = None
        self._excluded_files.clear()

        # Collect image files
        image_files = []
        for f in sorted(glob.glob('{}/*'.format(self.input_path))):
            if Path(f).suffix.lower() in CV2_FILETYPES:
                image_files.append(f)

        if not image_files:
            self.gallery_placeholder.show()
            self.gallery_scroll.hide()
            self.gallery_count_label.hide()
            return

        self.gallery_placeholder.hide()
        self.gallery_scroll.show()
        self.gallery_count_label.show()
        self.gallery_count_label.setText("{} IMAGES".format(len(image_files)))

        cols = 4
        for idx, file_path in enumerate(image_files):
            row = idx // cols
            col = idx % cols

            thumb = ClickableThumbnail(file_path)
            thumb.clicked.connect(self.on_thumbnail_clicked)
            thumb.removed.connect(self.on_thumbnail_removed)
            self.gallery_grid.addWidget(thumb, row, col)
            self._thumbnail_widgets[file_path] = thumb

        self.right_stack.setCurrentIndex(0)

        # Start background thread to load thumbnails
        self._thumb_loader_thread = ThumbnailLoaderThread(image_files)
        self._thumb_loader_thread.thumbnail_ready.connect(self._on_thumbnail_loaded)
        self._thumb_loader_thread.start()

    def _on_thumbnail_loaded(self, file_path, qimage):
        """Slot: set the loaded thumbnail pixmap on its widget."""
        thumb = self._thumbnail_widgets.get(file_path)
        if thumb is not None:
            pixmap = QtGui.QPixmap.fromImage(qimage)
            thumb.set_pixmap(pixmap)

    def on_thumbnail_clicked(self, file_path):
        """Select the clicked thumbnail (highlight it)."""
        # Deselect previous
        if self._selected_file and self._selected_file in self._thumbnail_widgets:
            self._thumbnail_widgets[self._selected_file].set_selected(False)

        # Select new
        self._selected_file = file_path
        if file_path in self._thumbnail_widgets:
            self._thumbnail_widgets[file_path].set_selected(True)

    def on_thumbnail_removed(self, file_path):
        """Remove a thumbnail from the gallery and reflow the grid."""
        self._excluded_files.add(file_path)

        # Clear selection if the removed image was selected
        if self._selected_file == file_path:
            self._selected_file = None

        # Remove the widget
        thumb = self._thumbnail_widgets.pop(file_path, None)
        if thumb is not None:
            self.gallery_grid.removeWidget(thumb)
            thumb.deleteLater()

        # Update count label
        remaining = len(self._thumbnail_widgets)
        if remaining == 0:
            self.gallery_placeholder.show()
            self.gallery_scroll.hide()
            self.gallery_count_label.hide()
            return

        self.gallery_count_label.setText("{} IMAGES".format(remaining))

        # Reflow: detach all widgets from grid, then re-add in order
        widgets = []
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            w = item.widget()
            if w:
                widgets.append(w)

        cols = 4
        for idx, w in enumerate(widgets):
            self.gallery_grid.addWidget(w, idx // cols, idx % cols)

    def preview_selected(self):
        """Preview button handler — previews the selected or first image."""
        file_path = self._selected_file

        # If no image selected, use the first non-excluded image in the input folder
        if file_path is None:
            try:
                for f in sorted(glob.glob('{}/*'.format(self.input_path))):
                    if Path(f).suffix.lower() in CV2_FILETYPES and f not in self._excluded_files:
                        file_path = f
                        break
            except AttributeError:
                self.error_popup(self.warning_folders)
                return

        if file_path is None:
            self.error_popup(self.warning_no_file)
            return

        self.preview_single(file_path)

    def _facecrop_cache_key(self, kwargs, bg_color):
        """Build a hashable key from FaceCrop settings for cache comparison."""
        return (tuple(sorted(kwargs.items())), bg_color)

    def _get_or_invalidate_cache(self, kwargs, bg_color):
        """Return cached FaceCrop if settings match, otherwise invalidate cache."""
        key = self._facecrop_cache_key(kwargs, bg_color)
        if self._cached_facecrop is not None and self._cached_facecrop_key == key:
            return self._cached_facecrop
        # Settings changed — close old instance
        if self._cached_facecrop is not None:
            self._cached_facecrop.close()
            self._cached_facecrop = None
            self._cached_facecrop_key = None
        return None

    def preview_single(self, file_path):
        """Run face crop on a single file asynchronously and display the result."""
        # Ignore if a preview is already running
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return

        try:
            kwargs = self._get_facecrop_kwargs()
        except ValueError:
            self.error_popup(self.warning_values)
            return

        bg_color = None
        if self.checkbox_bg.isChecked():
            c = self._selected_bg_color
            bg_color = (c.red(), c.green(), c.blue())

        cached = self._get_or_invalidate_cache(kwargs, bg_color)

        # Show spinner and disable buttons
        self._spinner.start(self._text_processing_preview)
        self.preview_button.setEnabled(False)
        self.crop_button.setEnabled(False)

        self._preview_worker = PreviewWorkerThread(file_path, kwargs, bg_color,
                                                   cached_facecrop=cached)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_finished(self, result, file_path, face_crop):
        """Called when async preview completes."""
        self._spinner.stop()
        self.preview_button.setEnabled(True)
        self.crop_button.setEnabled(True)

        # Cache the FaceCrop instance for reuse
        if face_crop is not None:
            bg_color = None
            if self.checkbox_bg.isChecked():
                c = self._selected_bg_color
                bg_color = (c.red(), c.green(), c.blue())
            try:
                kwargs = self._get_facecrop_kwargs()
                self._cached_facecrop = face_crop
                self._cached_facecrop_key = self._facecrop_cache_key(kwargs, bg_color)
            except ValueError:
                pass

        if result is None:
            self.preview_image.setText(self.warning_no_face)
            self.preview_image.setStyleSheet(
                "color: #666; font-size: 12pt; background-color: #141414; border: 1px solid #333; border-radius: 4px;"
            )
            self.preview_filename.setText(Path(file_path).name)
            self.right_stack.setCurrentIndex(1)
            return

        pixMap = self._pil_to_pixmap(result)
        self.preview_image.setStyleSheet("")  # reset any text styling
        self.preview_image.setPixmap(pixMap.scaled(
            self.preview_image.width(),
            self.preview_image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.preview_filename.setText(Path(file_path).name)
        self.right_stack.setCurrentIndex(1)

    def _on_preview_error(self, error_type, error_message):
        """Called when async preview fails."""
        self._spinner.stop()
        self.preview_button.setEnabled(True)
        self.crop_button.setEnabled(True)

        if error_type == "init":
            self.error_popup("{}\n\n{}".format(self.warning_init, error_message))
        else:
            self.error_popup("Crop failed: {}".format(error_message))

    def _pil_to_pixmap(self, pil_image):
        """Convert a PIL Image to QPixmap without using PIL.ImageQt."""
        if pil_image.mode == 'RGBA':
            fmt = QtGui.QImage.Format_RGBA8888
            data = pil_image.tobytes('raw', 'RGBA')
            bpl = 4 * pil_image.width
        else:
            pil_image = pil_image.convert('RGB')
            fmt = QtGui.QImage.Format_RGB888
            data = pil_image.tobytes('raw', 'RGB')
            bpl = 3 * pil_image.width
        qimg = QtGui.QImage(data, pil_image.width, pil_image.height, bpl, fmt)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _on_bg_toggled(self, checked):
        self.bg_color_button.setEnabled(checked)

    def _on_crop_mode_changed(self, index):
        self.crop_mode_stack.setCurrentIndex(index)

    def _on_ratio_preset_changed(self, text):
        is_custom = (text == "Custom")
        self.ratio_custom_w_label.setVisible(is_custom)
        self.ratio_custom_w_input.setVisible(is_custom)
        self.ratio_custom_h_label.setVisible(is_custom)
        self.ratio_custom_h_input.setVisible(is_custom)

    def _get_aspect_ratio(self):
        text = self.ratio_combo.currentText()
        if text == "Custom":
            w = max(1, int(self.ratio_custom_w_input.text()))
            h = max(1, int(self.ratio_custom_h_input.text()))
            return (w, h)
        parts = text.split(":")
        return (int(parts[0]), int(parts[1]))

    def _get_facecrop_kwargs(self):
        """Gather current UI values into kwargs dict for FaceCrop constructor.
        Raises ValueError if any input is invalid."""
        height_asy = float(self.height_asy_input.text())
        width_asy = float(self.width_asy_input.text())
        tag = str(self.tag_input.text())

        mode_index = self.crop_mode_combo.currentIndex()
        mode_map = {0: 'percentage', 1: 'aspect_ratio', 2: 'custom_pixels'}
        mode = mode_map[mode_index]

        kwargs = {
            'height_asy': height_asy,
            'width_asy': width_asy,
            'tag': tag,
            'mode': mode,
            'height': 0,
            'width': 0,
            'aspect_ratio': (1, 1),
            'padding_multiplier': 2.5,
            'custom_width_px': 600,
            'custom_height_px': 800,
        }

        if mode == 'percentage':
            kwargs['height'] = float(self.height_input.text())
            kwargs['width'] = float(self.width_input.text())
        elif mode == 'aspect_ratio':
            kwargs['aspect_ratio'] = self._get_aspect_ratio()
            kwargs['padding_multiplier'] = float(self.padding_input.text())
        elif mode == 'custom_pixels':
            kwargs['custom_width_px'] = int(self.px_width_input.text())
            kwargs['custom_height_px'] = int(self.px_height_input.text())

        return kwargs

    def _pick_bg_color(self):
        color = QtWidgets.QColorDialog.getColor(self._selected_bg_color, None, "Select Background Color")
        if color.isValid():
            self._selected_bg_color = color
            self._update_color_button()

    def _update_color_button(self):
        self.bg_color_button.setStyleSheet(
            "QPushButton#bg_color_button {{ background-color: {}; }}".format(self._selected_bg_color.name())
        )

    def crop_all(self):
        """Batch crop all images."""
        # Ignore if crop init is already running
        if self._crop_init_worker is not None and self._crop_init_worker.isRunning():
            return

        try:
            kwargs = self._get_facecrop_kwargs()
        except ValueError:
            self.error_popup(self.warning_values)
            return

        bool_folder = self.checkbox_folder.isChecked()
        bool_face_count = self.checkbox_count.isChecked()

        bg_color = None
        if self.checkbox_bg.isChecked():
            c = self._selected_bg_color
            bg_color = (c.red(), c.green(), c.blue())

        #making sure input and output path were defined by the user
        try:
            self.input_path
            self.output_path
            bool_folders = [os.path.isdir(i) for i in self.paths]
            subfolders = any(bool_folders)
        except Exception:
            self.error_popup(self.warning_folders)
            return

        #saves preferences if app is autorised to write file
        if autorisation:
            self.update_params(parameters, param_file_path)

        # Store batch params for callback
        self._batch_params = {
            'bool_folder': bool_folder,
            'bool_face_count': bool_face_count,
            'subfolders': subfolders,
            'directories': list(compress(self.paths, bool_folders)) if subfolders else None,
        }

        # Show spinner during model initialization
        cached = self._get_or_invalidate_cache(kwargs, bg_color)
        self._spinner.start(self._text_loading_model)
        self.crop_button.setEnabled(False)
        self.preview_button.setEnabled(False)

        self._crop_init_worker = CropInitWorkerThread(kwargs, bg_color, pyqt_ui=self,
                                                      cached_facecrop=cached)
        self._crop_init_worker.finished.connect(self._on_crop_init_done)
        self._crop_init_worker.error.connect(self._on_crop_init_error)
        self._crop_init_worker.start()

    def _on_crop_init_done(self, face_crop):
        """Called when FaceCrop is initialized for batch cropping."""
        self._spinner.stop()
        self.crop_button.setEnabled(True)
        self.preview_button.setEnabled(True)

        params = self._batch_params

        if params['subfolders']:
            directories = params['directories']
            bar_length = sum([len(next(os.walk(dir_))[2]) for dir_ in directories])
            self.progress_bar(bar_length)
            for directory in directories:
                if self.progress.wasCanceled():
                    break
                face_crop.crop_save(directory, self.output_path,
                                    bool_folder=params['bool_folder'],
                                    bool_face_count=params['bool_face_count'],
                                    excluded_files=self._excluded_files)
        else:
            bar_length = len(next(os.walk(self.input_path))[2])
            self.progress_bar(bar_length)
            face_crop.crop_save(self.input_path, self.output_path,
                                bool_folder=params['bool_folder'],
                                bool_face_count=params['bool_face_count'],
                                excluded_files=self._excluded_files)

        # Cache the FaceCrop instance for reuse (don't close it)
        bg_color = None
        if self.checkbox_bg.isChecked():
            c = self._selected_bg_color
            bg_color = (c.red(), c.green(), c.blue())
        try:
            kwargs = self._get_facecrop_kwargs()
            self._cached_facecrop = face_crop
            self._cached_facecrop_key = self._facecrop_cache_key(kwargs, bg_color)
        except ValueError:
            face_crop.close()

        # Open the output folder after cropping is finished
        if sys.platform == 'win32':
            os.startfile(self.output_path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', self.output_path])
        else:
            subprocess.Popen(['xdg-open', self.output_path])

    def _on_crop_init_error(self, error_message):
        """Called when FaceCrop initialization fails for batch cropping."""
        self._spinner.stop()
        self.crop_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.error_popup("{}\n\n{}".format(self.warning_init, error_message))

    def error_popup(self, text):
         self.msg = QtWidgets.QMessageBox()
         self.msg.setWindowTitle(self.warning_title)
         self.msg.setText(text)
         self.msg.setIcon(QtWidgets.QMessageBox.Critical)
         self.msg.exec_()

    def progress_bar(self, length):
        self.progress = QtWidgets.QProgressDialog("Processing...", "Cancel", 0, length)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setWindowTitle("Cropping in progress")
        self.progress.setFixedSize(500, 100)
        self.progress.setMinimumDuration(100)

    def update_params(self, data, file_name):
        data['width'] = self.width_input.text()
        data['height'] = self.height_input.text()
        data['width_asy'] = self.width_asy_input.text()
        data['height_asy'] = self.height_asy_input.text()
        data['tag'] = self.tag_input.text()
        data['folder_option'] = self.checkbox_folder.isChecked()
        data['single_face_option'] = self.checkbox_count.isChecked()
        data['crop_mode'] = self.crop_mode_combo.currentIndex()
        data['ratio_preset'] = self.ratio_combo.currentText()
        data['ratio_custom_w'] = self.ratio_custom_w_input.text()
        data['ratio_custom_h'] = self.ratio_custom_h_input.text()
        data['padding_multiplier'] = self.padding_input.text()
        data['custom_width_px'] = self.px_width_input.text()
        data['custom_height_px'] = self.px_height_input.text()

        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)


def main(app_language):
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    # Set application icon
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public', 'logo.png')
    if os.path.exists(_logo_path):
        app.setWindowIcon(QtGui.QIcon(_logo_path))

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(app_language)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
