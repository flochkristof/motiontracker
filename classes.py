from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QDialogButtonBox,
    QPushButton,
    QVBoxLayout,
    QLineEdit,
    QWidget,
)
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal


class ObjectToTrack:
    def __init__(self, name, point, rectangle):
        self.name = name
        self.point = point
        self.rectangle = rectangle


"""
class PointPickerWidget(QWidget):
    def __init__(self):
        self.NameLNE = QLineEdit()
        self.NameLNE.setMaxLength(10)
        self.AddNewObjectBTN = QPushButton()
        self.AddNewObjectBTN.setText("Add new")
        self.PickPointBTN = QPushButton()
        self.PickPointBTN.setText("Pick point")
        self.PickRectBTN = QPushButton()
        self.PickRectBTN.setText("Pick rectangle")"""


class VideoLabel(QLabel):
    press = pyqtSignal(float, float)
    moving = pyqtSignal(float, float)
    release = pyqtSignal(float, float)

    def __init__(self, parent=None):
        self.press_pos = None
        self.current_pos = None
        super(VideoLabel, self).__init__(parent)

    def mousePressEvent(self, ev: QMouseEvent):
        x_label, y_label, = ev.x(), ev.y()

        if self.pixmap():
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            width = pixmap_size.width()
            height = pixmap_size.height()

            x_0 = int((label_size.width() - width) / 2)
            y_0 = int((label_size.height() - height) / 2)

            if (
                x_label >= x_0
                and x_label < (x_0 + width)
                and y_label >= y_0
                and y_label < (y_0 + height)
            ):
                x_rel = (x_label - x_0 - width / 2) / width
                y_rel = (y_label - y_0 - height / 2) / height
                self.press_pos = (x_rel, y_rel)
                self.press.emit(x_rel, y_rel)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        x_label, y_label, = ev.x(), ev.y()

        if self.pixmap():
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            width = pixmap_size.width()
            height = pixmap_size.height()

            x_0 = int((label_size.width() - width) / 2)
            y_0 = int((label_size.height() - height) / 2)

            if (
                x_label >= x_0
                and x_label < (x_0 + width)
                and y_label >= y_0
                and y_label < (y_0 + height)
            ):
                x_rel = (x_label - x_0 - width / 2) / width
                y_rel = (y_label - y_0 - height / 2) / height
                self.moving.emit(x_rel, y_rel)
                self.current_pos = (x_rel, y_rel)
        super().mousePressEvent(ev)

    """def mouseReleaseEvent(self, ev: QMouseEvent):
        x_label, y_label, = ev.x(), ev.y()

        if self.pixmap():
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            width = pixmap_size.width()
            height = pixmap_size.height()

            x_0 = int((label_size.width() - width) / 2)
            y_0 = int((label_size.height() - height) / 2)

            if (
                x_label >= x_0
                and x_label < (x_0 + width)
                and y_label >= y_0
                and y_label < (y_0 + height)
            ):
                x_rel = (x_label - x_0 - width / 2) / width
                y_rel = (y_label - y_0 - height / 2) / height
                self.release.emit(x_rel, y_rel)
                print("released")
        super().mousePressEvent(ev)"""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# class TimeInputDialog(QDialog):
#    def __init__(self, *args, **kwargs):
#        super(TimeInputDialog, self).__init__(*args, **kwargs)
#        # Input QLineEdit
#        #TimeLineEdit = QLineEdit()
#        #TimeLineEdit.setInputMask("")
#
#        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
#        buttonBox = QDialogButtonBox(QBtn)
#        buttonBox.accepted.connect(self.accept)
#        buttonBox.rejected.connect(self.reject)
#        layout = QVBoxLayout()
#        layout.addWidget(buttonBox)
#        self.setLayout(self.layout)
