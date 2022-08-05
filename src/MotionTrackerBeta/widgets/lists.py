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

from PyQt5.QtWidgets import QListWidget,QMenu
from PyQt5.QtGui import QCursor

from PyQt5.QtCore import pyqtSignal
from MotionTrackerBeta.functions.helper import *
from MotionTrackerBeta.classes.classes import *


class RotationListWidget(QListWidget):
    """Widget that displays the created rotations"""

    # create signal
    delete = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize"""

        # call parent function
        super(RotationListWidget, self).__init__(parent)

        # connect signal
        self.itemClicked.connect(self.listItemMenu)

    def listItemMenu(self, item):
        """Open menu"""
        menu = QMenu()
        menu.addAction("Delete", lambda: self.delete.emit(item.text()))
        menu.exec_(QCursor.pos())
        menu.deleteLater()



class ObjectListWidget(QListWidget):
    """Widget that displays the objects created by the user"""

    # create signals
    delete = pyqtSignal(str)
    changeVisibility = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize"""

        # call parent function
        super(ObjectListWidget, self).__init__(parent)

        # connect signal
        self.itemClicked.connect(self.listItemMenu)

    def listItemMenu(self, item):
        """Opens the menu"""
        menu = QMenu()
        menu.addAction("Show/Hide", lambda: self.changeVisibility.emit(item.text()))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.delete.emit(item.text()))
        menu.exec_(QCursor.pos())
        menu.deleteLater()
