from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QDialogButtonBox,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QLineEdit,
    QWidget,
    QAction,
)
from PyQt5.QtGui import QMouseEvent, QCursor
from PyQt5.QtCore import Qt, pyqtSignal
import math


class ObjectToTrack:
    def __init__(self, name, point, rectangle, visible=True):
        self.name = name
        self.point = point
        self.rectangle = rectangle
        self.visible = visible


class Ruler:
    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.mm = None
        self.mm_per_pix = None
        self.rdy = False

    def setP0(self, x, y):
        self.x0 = int(x)
        self.y0 = int(y)
        self.calculate()

    def setP1(self, x, y):
        self.x1 = int(x)
        self.y1 = int(y)
        self.calculate()

    def clear(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

    def calculate(self):
        if (
            self.x0 is not None
            and self.y0 is not None
            and self.x1 is not None
            and self.y1 is not None
            and self.mm is not None
        ):
            pix = math.sqrt(
                math.pow(self.x1 - self.x0, 2) + math.pow(self.y1 - self.y0, 2)
            )
            self.mm_per_pix = self.mm / pix
            self.rdy = True

    def displayable(self):
        if (
            self.x0 is not None
            and self.y0 is not None
            and self.x1 is not None
            and self.y1 is not None
        ):
            return True
        else:
            return False

    def reset(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.mm = None
        self.mm_per_pix = None


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

    def mouseReleaseEvent(self, ev: QMouseEvent):
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
        super().mousePressEvent(ev)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ObjectListWidget(QListWidget):
    delete = pyqtSignal(str)
    changeVisibility = pyqtSignal(str)

    def __init__(self, parent=None):
        super(ObjectListWidget, self).__init__(parent)
        self.itemClicked.connect(self.listItemMenu)

    def listItemMenu(self, item):
        menu = QMenu()
        menu.addAction("Show/Hide", lambda: self.changeVisibility.emit(item.text()))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.delete.emit(item.text()))
        menu.exec_(QCursor.pos())
