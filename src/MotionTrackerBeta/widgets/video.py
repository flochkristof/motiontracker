# Copyright 2022 Kristof Floch
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMouseEvent,QWheelEvent
from PyQt5.QtCore import pyqtSignal

from MotionTrackerBeta.functions.helper import *



class VideoLabel(QLabel):
    """Label to display the frames from OpenCV"""

    # create signals
    press = pyqtSignal(float, float)
    moving = pyqtSignal(float, float)
    release = pyqtSignal(float, float)
    wheel = pyqtSignal(float)

    def __init__(self, parent=None):
        """Intitialization"""
        self.press_pos = None
        self.current_pos = None
        super(VideoLabel, self).__init__(parent)

    def wheelEvent(self, a0: QWheelEvent):
        """Handles wheel event, emits signal with zoom parameter"""
        if a0.angleDelta().y() > 0:
            self.wheel.emit(-0.1)
        else:
            self.wheel.emit(0.1)
        return super().wheelEvent(a0)

    def mousePressEvent(self, ev: QMouseEvent):
        """Handles mouse press event, emits coordinates"""
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
        """Handles mouse movement, emits coordinates"""
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
        """Handles mouse release, emits coordinates"""
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