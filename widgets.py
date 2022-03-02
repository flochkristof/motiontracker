from sqlite3 import Timestamp
import turtle
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QDialogButtonBox,
    QProgressBar,
    QSpacerItem,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QLineEdit,
    QWidget,
    QAction,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QFrame,
)
from PyQt5.QtGui import QMouseEvent, QCursor, QWheelEvent, QIntValidator
from PyQt5.QtCore import QLine, QThread, Qt, pyqtSignal, QEventLoop, QPoint
import math
import cv2
import numpy as np
from funcs import *
from classes import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import time  # only to measure performance


class ExportingThread(QThread):
    progressChanged = pyqtSignal(int)
    success = pyqtSignal()
    error_occured = pyqtSignal(str)

    def __init__(
        self,
        camera,
        objects_to_track,
        start,
        stop,
        filename,
        fps,
        box_bool,
        point_bool,
        trajectory_lenght,
    ):
        self.camera = camera
        self.objects = objects_to_track
        self.section_start = start
        self.section_stop = stop
        self.filename = filename
        self.fps = fps
        self.is_running = True
        self.box_bool = box_bool
        self.point_bool = point_bool
        self.trajectory_length = trajectory_lenght
        super(ExportingThread, self).__init__()

    def cancel(self):
        self.is_running = False

    def run(self):
        # init writer
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))

        # goto start
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)
        for i in range(int(self.section_stop - self.section_start)):
            pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.camera.read()
            if not ret:
                self.is_running = False
                self.error_occured.emit("Unable to read video frame!")
                break
            frame = display_objects(
                frame,
                pos,
                self.section_start,
                self.section_stop,
                self.objects,
                self.box_bool,
                self.point_bool,
                self.trajectory_length,
            )
            writer.write(frame)
            self.progressChanged.emit(
                int(pos / (self.section_stop - self.section_start) * 100)
            )
        if self.is_running:
            self.success.emit()


class TrackingThread(QThread):
    """QThread class responsible for running the tracking algorithm"""

    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success = pyqtSignal()
    rotation_calculated = pyqtSignal(Rotation)
    error_occured = pyqtSignal(str)

    def __init__(
        self,
        objects_to_track,
        camera,
        start,
        stop,
        tracker_type,
        size,
        fps,
        filter_settings,
        derivative_settings,
        timestamp,
        roi_rect,
    ):
        self.objects_to_track = objects_to_track
        self.camera = camera
        self.section_start = start
        self.section_stop = stop
        self.tracker_type = tracker_type
        self.timestamp = timestamp
        self.roi_rect = roi_rect
        self.size = size
        self.fps = fps
        self.filter_settings = filter_settings
        self.derivative_settings = derivative_settings
        self.progress = "0"
        self.is_running = True
        if roi_rect is None:
            self.roi_rect = (0, 0)
        else:
            self.roi_rect = roi_rect
        super(TrackingThread, self).__init__()

    def cancel(self):
        self.is_running = False

    def run(self):
        start_time = time.time()
        for j in range(len(self.objects_to_track)):
            M = self.objects_to_track[j]
            # emit the name of the tracked object
            self.newObject.emit("Tracking object: " + M.name + "...")

            # reset previous data
            M.reset_data()
            self.timestamp.clear()

            # creating the tracker
            if self.tracker_type == "BOOSTING":
                tracker = cv2.legacy.TrackerBoosting_create()
            if self.tracker_type == "MIL":
                tracker = cv2.legacy.TrackerMIL_create()
            if self.tracker_type == "KCF":
                tracker = cv2.TrackerKCF_create()
            if self.tracker_type == "TLD":
                tracker = cv2.legacy.TrackerTLD_create()
            if self.tracker_type == "MEDIANFLOW":
                tracker = cv2.legacy.TrackerMedianFlow_create()
            if self.tracker_type == "GOTURN":
                tracker = cv2.TrackerGOTURN_create()
            if self.tracker_type == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            if self.tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()

            # set camera to start frame, get the fps
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start - 1)

            # initialize tracker
            ret, frame = self.camera.read()
            if not ret:
                self.error_occured.emit("Unable to read video frame!")
                return

            if len(self.roi_rect) == 4:
                frame = crop_roi(frame, self.roi_rect)

            tracker.init(frame, rect2cropped(M.rectangle, self.roi_rect))

            # for the calculation of the point
            dy = (M.point[1] - M.rectangle[1]) / M.rectangle[3]
            dx = (M.point[0] - M.rectangle[0]) / M.rectangle[2]

            # for zoom
            w0 = M.rectangle[2]
            h0 = M.rectangle[3]

            # tracking
            for i in range(int(self.section_stop - self.section_start)):
                # read the next frame
                ret, frame = self.camera.read()
                if not ret:
                    self.error_occured.emit("Unable to read video frame!")
                    self.is_running = False
                    break

                if len(self.roi_rect) == 4:
                    frame = crop_roi(frame, self.roi_rect)

                # update the tracker
                ret, roi_box = tracker.update(frame)
                if ret:
                    # traditional tracking
                    x, y, w, h = roi_box
                    M.rectangle_path.append(
                        (self.roi_rect[0] + x, self.roi_rect[1] + y, w, h)
                    )
                    # M.rectangle_path.append(roi_box)
                    M.point_path.append(
                        (
                            self.roi_rect[0] + roi_box[0] + dx * roi_box[2],
                            self.roi_rect[1] + roi_box[1] + dy * roi_box[3],
                        )
                    )

                    # change of size
                    if self.size:
                        M.size_change.append((roi_box[2] / w0 + roi_box[3] / h0) / 2)

                    if j == 0:
                        self.timestamp.append((i) / self.fps)
                    # progress
                    self.progress = math.ceil(
                        i / (self.section_stop - self.section_start) * 100
                    )
                    self.progressChanged.emit(self.progress)
                    # self.status = f"Tracking ... {self.progress}"
                else:
                    self.error_occured.emit("Tracking failed!")
                    self.is_running = False

                if not self.is_running:
                    M.rectangle_path = []
                    break

            if not self.is_running:
                break

            ### POST-PROCESSING START ###
            ## FILTERS ##
            self.newObject.emit("Post-processing... applying filters")
            if self.filter_settings["filter"] == "None":
                # no filter applied, only converting the data
                M.position = list2np(M.point_path)
            elif self.filter_settings["filter"] == "Gaussian":
                M.position = gaussian(
                    M.point_path,
                    self.filter_settings["window"],
                    self.filter_settings["sigma"],
                )
            elif self.filter_settings["filter"] == "Moving AVG":
                M.position = moving_avg(M.point_path, self.filter_settings["window"])
                ts = self.timestamp[0 : len(M.position)]
                self.timestamp.clear()
                self.timestamp.extend(ts)
                print(f"timestamp:{len(self.timestamp)}; posi:{len(M.position)}")

            elif self.filter_settings["filter"] == "SG":
                M.position = savgol(
                    M.point_path,
                    self.filter_settings["window"],
                    self.filter_settings["pol"],
                )
            print(M.position)
            ## DERIVATIVES ##
            # if self.filter
            # M.velocity = M.position
            # M.acceleration = M.position

            if self.derivative_settings["derivative"] == "SG":
                M.velocity = savgol_diff(
                    M.position,
                    self.derivative_settings["window"],
                    self.derivative_settings["pol"],
                    1,
                    self.fps,
                )
                M.acceleration = savgol_diff(
                    M.position,
                    self.derivative_settings["window"],
                    self.derivative_settings["pol"],
                    2,
                    self.fps,
                )
            elif self.derivative_settings["derivative"] == "FINDIFF":
                M.velocity = fin_diff(
                    M.position, self.derivative_settings["acc"], 1, self.fps,
                )
                M.acceleration = fin_diff(
                    M.position, self.derivative_settings["acc"], 2, self.fps,
                )

        ### end of object-by-object for loop

        # rotation tracking
        # if len(self.rotation_endpoints) > 0:
        #    self.newObject.emit("Calculating rotation...")
        #    p1_index = next(
        #        (
        #            i
        #            for i, item in enumerate(self.objects_to_track)
        #            if item.name == self.rotation_endpoints[0]
        #        ),
        #        -1,
        #    )
        #    p2_index = next(
        #        (
        #            i
        #            for i, item in enumerate(self.objects_to_track)
        #            if item.name == self.rotation_endpoints[1]
        #        ),
        #        -1,
        #    )
        #    if p1_index != -1 and p2_index != -1:
        #        P1 = self.objects_to_track[p1_index]
        #        P2 = self.objects_to_track[p2_index]
        #        self.newObject.emit("Rotation between :" + P1.name + " and " + P2.name)
        #        R = Rotation(P1, P2)
        #        R.calculate()
        #        self.rotation_calculated.emit(R)
        #
        #    else:
        #        self.error_occured.emit(
        #            "Uable to calculate rotation with only one point!"
        #        )
        #        self.is_running = False

        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)
        if self.is_running:
            self.success.emit()
        print(f"finished in {time.time()-start_time}")


class VideoLabel(QLabel):
    """Label to display the frames from OpenCV"""

    press = pyqtSignal(float, float)
    moving = pyqtSignal(float, float)
    release = pyqtSignal(float, float)
    wheel = pyqtSignal(float)

    def __init__(self, parent=None):
        self.press_pos = None
        self.current_pos = None
        super(VideoLabel, self).__init__(parent)

    def wheelEvent(self, a0: QWheelEvent):
        if a0.angleDelta().y() > 0:
            self.wheel.emit(-0.1)
        else:
            self.wheel.emit(0.1)
        return super().wheelEvent(a0)

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


class ObjectListWidget(QListWidget):
    """Widget that displays the objects created by the user"""

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
        menu.deleteLater()


class RotationListWidget(QListWidget):
    """Widget that displays the rotations"""

    delete = pyqtSignal(str)

    def __init__(self, parent=None):
        super(RotationListWidget, self).__init__(parent)
        self.itemClicked.connect(self.listItemMenu)

    def listItemMenu(self, item):
        menu = QMenu()
        # menu.addAction("Show/Hide", lambda: self.changeVisibility.emit(item.text()))
        # menu.addSeparator()
        menu.addAction("Delete", lambda: self.delete.emit(item.text()))
        menu.exec_(QCursor.pos())
        menu.deleteLater()

    pass


class TrackingSettings(QDialog):
    """Modal dialog where users can specify the tracking settings"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking details")
        self.setModal(True)

        #### Additional Features ###
        # self.rotationCHB = QCheckBox("Track rotation")
        # self.rotationCHB.stateChanged.connect(self.openRotationSettings)
        # self.sizeCHB = QCheckBox("Track size change")
        # self.sizeCHB.stateChanged.connect(self.sizeMode)
        #
        # featureLayout = QVBoxLayout()
        # featureLayout.addWidget(self.rotationCHB)
        # featureLayout.addWidget(self.sizeCHB)
        #
        # featureGB = QGroupBox("Additional features to track")
        # featureGB.setLayout(featureLayout)
        #
        #### Tracking algorithm ###
        # self.algoritmCMB = QComboBox()
        # self.algoritmCMB.addItems(
        #    ["CSRT", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE"]
        # )
        #
        # self.zoomNotificationLBL = QLabel(
        #    "Only the CSRT algorithm is capable of handling the size change of an object!"
        # )
        # self.zoomNotificationLBL.setVisible(False)
        # algoLayout = QVBoxLayout()
        # algoLayout.addWidget(self.algoritmCMB)
        # algoLayout.addWidget(self.zoomNotificationLBL)
        #
        # algoGB = QGroupBox("Tracking algorithm")
        # algoGB.setLayout(algoLayout)

        ### Tracking settings ###
        algoLBL = QLabel("Tracking algorithm")
        self.algoritmCMB = QComboBox()
        self.algoritmCMB.addItems(
            ["CSRT", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE"]
        )

        ## Zoom settings ##
        self.sizeCHB = QCheckBox("Track size change")
        self.sizeCHB.stateChanged.connect(self.sizeMode)
        self.zoomNotificationLBL = QLabel(
            "Only the CSRT algorithm is capable of handling the size change of an object!"
        )
        self.zoomNotificationLBL.setVisible(False)
        self.zoomNotificationLBL.setWordWrap(True)

        ## organizing the layout
        algoLayout = QHBoxLayout()
        algoLayout.addWidget(algoLBL)
        algoLayout.addWidget(self.algoritmCMB)

        propLayout = QHBoxLayout()
        propLayout.addLayout(algoLayout)
        propLayout.addWidget(self.sizeCHB)
        # propLayout.addWidget(self.zoomNotificationLBL)

        topLayout = QVBoxLayout()
        topLayout.addLayout(propLayout)
        topLayout.addWidget(self.zoomNotificationLBL)

        ### Left side veritcal layout GroupBox ###
        topGB = QGroupBox("Tracking settings")
        topGB.setLayout(topLayout)

        ### Real FPS input ###
        fpsLBL = QLabel("Real FPS:")
        self.fpsLNE = QLineEdit()
        self.fpsLNE.setValidator(QIntValidator(0, 1000000))

        fpsLayout = QHBoxLayout()
        fpsLayout.addWidget(fpsLBL)
        fpsLayout.addWidget(self.fpsLNE)

        ### Filter ###
        filterLBL = QLabel("Filter")
        self.filterCMB = QComboBox()
        self.filterCMB.addItem("None")
        self.filterCMB.addItem("Gaussian")
        self.filterCMB.addItem("Moving AVG")
        self.filterCMB.addItem("Savitzky-Golay")
        self.filterCMB.currentTextChanged.connect(self.openFilterSettings)

        filterHLayout = QHBoxLayout()
        filterHLayout.addWidget(filterLBL)
        filterHLayout.addWidget(self.filterCMB)

        filterBTN = QPushButton("Filter settings")
        filterBTN.clicked.connect(self.openFilterSettings)

        filterVLayout = QVBoxLayout()
        filterVLayout.addLayout(filterHLayout)
        filterVLayout.addWidget(filterBTN)

        ### Derivative ###
        derivativeLBL = QLabel("Derivative:")
        self.derivativeCMB = QComboBox()
        self.derivativeCMB.addItem("LOESS-coefficients")
        self.derivativeCMB.addItem("Finite differences")
        self.derivativeCMB.currentTextChanged.connect(self.openDerivativeSettings)

        derivativeHLayout = QHBoxLayout()
        derivativeHLayout.addWidget(derivativeLBL)
        derivativeHLayout.addWidget(self.derivativeCMB)

        derivativeBTN = QPushButton("Derivative settings")
        derivativeBTN.clicked.connect(self.openDerivativeSettings)

        derivativeVLayout = QVBoxLayout()
        derivativeVLayout.addLayout(derivativeHLayout)
        derivativeVLayout.addWidget(derivativeBTN)

        # Botton layout and groupbox
        bottonLayout = QVBoxLayout()
        bottonLayout.addLayout(fpsLayout)

        processingLayout = QHBoxLayout()
        processingLayout.addLayout(filterVLayout)
        processingLayout.addLayout(derivativeVLayout)

        bottonLayout.addLayout(processingLayout)

        bottomGB = QGroupBox("Post-processing settings")
        bottomGB.setLayout(bottonLayout)

        # rightLayout = QVBoxLayout()
        # rightLayout.addLayout(fpsLayout)
        # rightLayout.addLayout(filterHLayout)
        # rightLayout.addWidget(filterBTN)
        # rightLayout.addLayout(derivativeHLayout)
        # rightLayout.addWidget(derivativeBTN)
        #
        # rightGB = QGroupBox("Post-processing settings")
        # rightGB.setLayout(rightLayout)
        #
        #### Settings horizontal layout ###
        # settingsLayout = QHBoxLayout()
        # settingsLayout.addWidget(leftGB)
        # settingsLayout.addWidget(rightGB)

        ### TRACK button and final layout
        trackBTN = QPushButton("Track")
        trackBTN.clicked.connect(self.accept)

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(topGB)
        mainlayout.addWidget(bottomGB)
        mainlayout.addWidget(trackBTN)

        self.setLayout(mainlayout)

        ### Rotation settings dialog ####
        self.rotationSettings = RotationSettings()
        self.filterSettings = FilterSettings()
        self.derivativeSettings = DerivativeSettings()

    # def openRotationSettings(self):
    #    if self.rotationCHB.isChecked():
    #        if self.rotationSettings.p1CMB.count() >= 2:
    #            self.rotationSettings.exec_()
    #        else:
    #            self.rotationCHB.setCheckState(Qt.Unchecked)
    #            msg = QMessageBox()
    #            msg.setWindowTitle("Not enough points!")
    #            msg.setText("You need at least two points for rotation tracking!")
    #            msg.setIcon(QMessageBox.Warning)
    #            msg.exec_()
    #            self.rotationCHB.setChecked(False)

    def openFilterSettings(self):
        filter_type = self.filterCMB.currentText()
        if filter_type != "None":
            self.filterSettings.setFilter(self.filterCMB.currentText())
            self.filterSettings.exec_()

    def openDerivativeSettings(self):
        self.derivativeSettings.setDerivative(self.derivativeCMB.currentText())
        self.derivativeSettings.exec_()

    def sizeMode(self):
        if self.sizeCHB.isChecked():
            self.zoomNotificationLBL.setVisible(True)
            self.algoritmCMB.setCurrentText("CSRT")
            self.algoritmCMB.setEditable(False)
            self.algoritmCMB.setEnabled(False)
        else:
            self.zoomNotificationLBL.setVisible(False)
            self.algoritmCMB.setEnabled(True)

    def tracker_type(self):
        return self.algoritmCMB.currentText()

    def size_change(self):
        return self.sizeCHB.isChecked()

    # def rotation(self):
    #    return self.rotationCHB.isChecked()

    def fps(self):
        return self.fpsLNE.text()


class RotationSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Rotation settings")
        self.setModal(True)
        # self.setAttribute(Qt.WA_DeleteOnClose)
        instuctionLBL = QLabel("Select two points for for tracking")
        self.warningLBL = QLabel("You must select two different points!")
        self.warningLBL.setVisible(False)
        okBTN = QPushButton("Save")
        okBTN.clicked.connect(self.validate)
        self.p1CMB = QComboBox()
        self.p2CMB = QComboBox()
        layout = QVBoxLayout()
        layout.addWidget(instuctionLBL)
        layout.addWidget(self.p1CMB)
        layout.addWidget(self.p2CMB)
        layout.addWidget(okBTN)
        layout.addWidget(self.warningLBL)
        self.setLayout(layout)

    def validate(self):
        if self.p1CMB.currentText() != self.p2CMB.currentText():
            self.accept()
        else:
            self.warningLBL.setVisible(True)

    def get_endpoints(self):
        return (self.p1CMB.currentText(), self.p2CMB.currentText())

    def set_params(self, objects):
        for obj in objects:
            self.p1CMB.addItem(str(obj))
            self.p2CMB.addItem(str(obj))


class FilterSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Filter settings")
        self.setModal(True)

        # GAUSS FILTER
        gwindowLBL = QLabel("Window")
        self.gwindowLNE = QLineEdit()
        self.gwindowLNE.setValidator(QIntValidator(0, 100))
        self.gwindowLNE.setText("5")
        gwindowLayout = QHBoxLayout()
        gwindowLayout.addWidget(gwindowLBL)
        gwindowLayout.addWidget(self.gwindowLNE)
        gsigmaLBL = QLabel("Sigma")
        self.gsigmaLNE = QLineEdit()
        self.gsigmaLNE.setValidator(QIntValidator(0, 10))
        self.gsigmaLNE.setText("3")
        gsigmaLayout = QHBoxLayout()
        gsigmaLayout.addWidget(gsigmaLBL)
        gsigmaLayout.addWidget(self.gsigmaLNE)
        gaussBTN = QPushButton("Set")
        gaussBTN.clicked.connect(self.accept)
        gaussLayout = QVBoxLayout()
        gaussLayout.addLayout(gwindowLayout)
        gaussLayout.addLayout(gsigmaLayout)
        gaussLayout.addWidget(gaussBTN)
        self.gaussGB = QGroupBox("Gauss")
        self.gaussGB.setLayout(gaussLayout)

        # MOVING AVERAGE
        mwindowLBL = QLabel("Window")
        self.mwindowLNE = QLineEdit()
        self.mwindowLNE.setValidator(QIntValidator(0, 100))
        self.mwindowLNE.setText("5")
        mwindowLayout = QHBoxLayout()
        mwindowLayout.addWidget(mwindowLBL)
        mwindowLayout.addWidget(self.mwindowLNE)
        mavgBTN = QPushButton("Set")
        mavgBTN.clicked.connect(self.accept)
        mavgLayout = QVBoxLayout()
        mavgLayout.addLayout(mwindowLayout)
        mavgLayout.addWidget(mavgBTN)
        self.mavgGB = QGroupBox("Moving AVG")
        self.mavgGB.setLayout(mavgLayout)

        # SAVITZKY-GOLAY
        sgwindowLBL = QLabel("Window")
        self.sgwindowLNE = QLineEdit()
        self.sgwindowLNE.setValidator(QIntValidator(0, 100))
        self.sgwindowLNE.setText("5")
        sgwindowLayout = QHBoxLayout()
        sgwindowLayout.addWidget(sgwindowLBL)
        sgwindowLayout.addWidget(self.sgwindowLNE)
        sgpolLBL = QLabel("Polynom")
        self.sgpolLNE = QLineEdit()
        self.sgpolLNE.setValidator(QIntValidator(0, 20))
        self.sgpolLNE.setText("3")
        sgpolLayout = QHBoxLayout()
        sgpolLayout.addWidget(sgpolLBL)
        sgpolLayout.addWidget(self.sgpolLNE)
        sgBTN = QPushButton("Set")
        sgBTN.clicked.connect(self.accept)
        sgLayout = QVBoxLayout()
        sgLayout.addLayout(sgwindowLayout)
        sgLayout.addLayout(sgpolLayout)
        sgLayout.addWidget(sgBTN)
        self.sgGB = QGroupBox("Savitzky-Golay")
        self.sgGB.setLayout(sgLayout)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.gaussGB)
        mainLayout.addWidget(self.mavgGB)
        mainLayout.addWidget(self.sgGB)

        self.setLayout(mainLayout)

    def setFilter(self, filter_type):
        if filter_type == "Gaussian":
            self.gaussGB.setVisible(True)
            self.mavgGB.setVisible(False)
            self.sgGB.setVisible(False)

        elif filter_type == "Moving AVG":
            self.gaussGB.setVisible(False)
            self.mavgGB.setVisible(True)
            self.sgGB.setVisible(False)

        elif filter_type == "Savitzky-Golay":
            self.gaussGB.setVisible(False)
            self.mavgGB.setVisible(False)
            self.sgGB.setVisible(True)


class DerivativeSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Derivative settings")
        self.setModal(True)

        # SAVITZKY-GOLAY
        sgwindowLBL = QLabel("Window")
        self.sgwindowLNE = QLineEdit()
        self.sgwindowLNE.setValidator(QIntValidator(0, 100))
        self.sgwindowLNE.setText("5")
        sgwindowLayout = QHBoxLayout()
        sgwindowLayout.addWidget(sgwindowLBL)
        sgwindowLayout.addWidget(self.sgwindowLNE)
        sgpolLBL = QLabel("Polynom")
        self.sgpolLNE = QLineEdit()
        self.sgpolLNE.setValidator(QIntValidator(0, 20))
        self.sgpolLNE.setText("3")
        sgpolLayout = QHBoxLayout()
        sgpolLayout.addWidget(sgpolLBL)
        sgpolLayout.addWidget(self.sgpolLNE)
        sgBTN = QPushButton("Set")
        sgBTN.clicked.connect(self.accept)
        sgLayout = QVBoxLayout()
        sgLayout.addLayout(sgwindowLayout)
        sgLayout.addLayout(sgpolLayout)
        sgLayout.addWidget(sgBTN)
        self.sgGB = QGroupBox("LOESS-coefficients")
        self.sgGB.setLayout(sgLayout)

        # FINTIE DIFFERENCES
        apprLBL = QLabel("Approximation order")
        self.apprLNE = QLineEdit()
        self.apprLNE.setValidator(QIntValidator(1, 20))
        self.apprLNE.setText("9")
        apprHLayout = QHBoxLayout()
        apprHLayout.addWidget(apprLBL)
        apprHLayout.addWidget(self.apprLNE)

        apprBTN = QPushButton("Set")
        apprBTN.clicked.connect(self.accept)

        apprLayout = QVBoxLayout()
        apprLayout.addLayout(apprHLayout)
        apprLayout.addWidget(apprBTN)

        self.apprGB = QGroupBox("Finite difference")
        self.apprGB.setLayout(apprLayout)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.sgGB)
        mainLayout.addWidget(self.apprGB)

        self.setLayout(mainLayout)

    def setDerivative(self, deriv_name):
        if deriv_name == "LOESS-coefficients":
            self.sgGB.setVisible(True)
            self.apprGB.setVisible(False)
        else:
            self.sgGB.setVisible(False)
            self.apprGB.setVisible(True)


class TrackingProgress(QDialog):
    """A modal dialog that shows the progress of the tracking"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Calculation in progress...")
        self.setModal(True)
        Layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setStyleSheet("text-align: center;")
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setMinimumWidth(400)
        cancelProgressBTN = QPushButton("Cancel")
        cancelProgressBTN.clicked.connect(self.rejected)
        Layout.addWidget(self.label, Qt.AlignCenter)
        Layout.addWidget(self.progressbar)
        Layout.addWidget(cancelProgressBTN)
        self.setLayout(Layout)

    def updateName(self, name):
        self.label.setText(name)

    def updateBar(self, value):
        self.progressbar.setValue(value)


class ExportDialog(QDialog):
    """Handles the exporting of the data collected by the trackers"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Export options")
        self.setModal(True)

        self.obj_checkboxes = []
        self.size_checkboxes = []
        self.rot_checkboxes = []
        self.parameters = {}

        # OBJECTS
        self.objNameLayout = QVBoxLayout()
        ObjGB = QGroupBox("Objects")
        ObjGB.setLayout(self.objNameLayout)

        # PROPERTY
        self.posRDB = QRadioButton("Position")
        self.velRDB = QRadioButton("Velocity")
        self.accRDB = QRadioButton("Acceleration")

        propBGP = QGroupBox("Proterty")
        propLayout = QVBoxLayout()
        propLayout.addWidget(self.posRDB)
        propLayout.addWidget(self.velRDB)
        propLayout.addWidget(self.accRDB)
        propBGP.setLayout(propLayout)

        # AXES
        self.xtRDB = QRadioButton("x(t)")
        self.ytRDB = QRadioButton("y(t)")

        axBGP = QGroupBox("Axis")
        axLayout = QVBoxLayout()
        axLayout.addWidget(self.xtRDB)
        axLayout.addWidget(self.ytRDB)
        axBGP.setLayout(axLayout)

        # UNITS
        self.mmRDB = QRadioButton("mm")
        self.mRDB = QRadioButton("m")
        self.pixRDB = QRadioButton("pixel")

        unitGB = QGroupBox("Units")
        unitLayout = QVBoxLayout()
        unitLayout.addWidget(self.mmRDB)
        unitLayout.addWidget(self.mRDB)
        unitLayout.addWidget(self.pixRDB)
        unitGB.setLayout(unitLayout)

        # ORGANIZING LAYOUT
        objHLayout = QHBoxLayout()
        objHLayout.addWidget(ObjGB)
        objHLayout.addWidget(propBGP)
        objHLayout.addWidget(axBGP)
        objHLayout.addWidget(unitGB)

        # EXPORT AND PLOT
        plotObjBTN = QPushButton("Plot")
        plotObjBTN.clicked.connect(self.plot_movement)
        exportObjBTN = QPushButton("Export")
        exportObjBTN.clicked.connect(self.export_movement)

        buttonHLayout = QHBoxLayout()
        buttonHLayout.addWidget(plotObjBTN)
        buttonHLayout.addWidget(exportObjBTN)

        objLBL = QLabel("Tracked objects")
        objLBL.setAlignment(Qt.AlignCenter)
        objLBL.setStyleSheet("font-weight: bold; font-size: 14px;")
        objLBL.setMaximumHeight(20)

        objVLayout = QVBoxLayout()
        objVLayout.addWidget(objLBL)
        objVLayout.addLayout(objHLayout)
        objVLayout.addLayout(buttonHLayout)

        self.objFrame = QFrame()
        self.objFrame.setLayout(objVLayout)

        ## ROTATION ##
        rotLBL = QLabel("Tracked rotation")
        rotLBL.setAlignment(Qt.AlignCenter)
        rotLBL.setStyleSheet("font-weight: bold; font-size: 14px;")
        rotLBL.setMaximumHeight(20)

        # ROTATION NAME
        self.rotNameLayout = QVBoxLayout()
        rotGB = QGroupBox("Rotation")
        rotGB.setLayout(self.rotNameLayout)

        # PROPERTY
        self.rot_posRDB = QRadioButton("Position")
        self.rot_velRDB = QRadioButton("Velocity")
        self.rot_accRDB = QRadioButton("Acceleration")

        rot_propBGP = QGroupBox("Property")
        rot_propLayout = QVBoxLayout()
        rot_propLayout.addWidget(self.rot_posRDB)
        rot_propLayout.addWidget(self.rot_velRDB)
        rot_propLayout.addWidget(self.rot_accRDB)
        rot_propBGP.setLayout(rot_propLayout)

        # UNITS
        self.degRDB = QRadioButton("degree")
        self.radRDB = QRadioButton("radian")

        rot_unitGB = QGroupBox("Units")
        rot_unitLayout = QVBoxLayout()
        rot_unitLayout.addWidget(self.degRDB)
        rot_unitLayout.addWidget(self.radRDB)
        rot_unitGB.setLayout(rot_unitLayout)

        # BUTTONS
        rotPlotBTN = QPushButton("Plot")
        rotPlotBTN.clicked.connect(self.plot_rotation)
        rotExportBTN = QPushButton("Export")
        rotExportBTN.clicked.connect(self.export_rotation)

        rotButtonLayout = QHBoxLayout()
        rotButtonLayout.addWidget(rotPlotBTN)
        rotButtonLayout.addWidget(rotExportBTN)

        rotHLayout = QHBoxLayout()
        rotHLayout.addWidget(rotGB)
        rotHLayout.addWidget(rot_propBGP)
        rotHLayout.addWidget(rot_unitGB)

        rotLayout = QVBoxLayout()
        rotLayout.addWidget(rotLBL)
        rotLayout.addLayout(rotHLayout)
        rotLayout.addLayout(rotButtonLayout)

        self.rotFrame = QFrame()
        self.rotFrame.setLayout(rotLayout)

        ## SIZE CHANGE ##
        sizeLBL = QLabel("Size change")
        sizeLBL.setAlignment(Qt.AlignCenter)
        sizeLBL.setStyleSheet("font-weight: bold; font-size: 14px;")
        sizeLBL.setMaximumHeight(20)

        self.sizeObjLayout = QVBoxLayout()
        sizeObjGB = QGroupBox("Objects")
        sizeObjGB.setLayout(self.sizeObjLayout)

        sizePlotBTN = QPushButton("Plot")
        sizePlotBTN.clicked.connect(self.plot_size)
        sizeExportBTN = QPushButton("Export")
        sizeExportBTN.clicked.connect(self.export_size)

        sizeBTNLayout = QHBoxLayout()
        sizeBTNLayout.addWidget(sizePlotBTN)
        sizeBTNLayout.addWidget(sizeExportBTN)

        sizeVLayout = QVBoxLayout()
        sizeVLayout.addWidget(sizeLBL)
        sizeVLayout.addWidget(sizeObjGB)
        sizeVLayout.addLayout(sizeBTNLayout)

        self.sizeFrame = QFrame()
        self.sizeFrame.setLayout(sizeVLayout)

        # NO ITEM WARNING
        self.warningLBL = QLabel("No tracked object to plot or export!")
        self.warningLBL.setAlignment(Qt.AlignCenter)
        self.warningLBL.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.warningLBL.setVisible(False)

        ## OVERALL LAYOUTS ##
        self.Layout = QVBoxLayout()
        self.Layout.addWidget(self.objFrame)

        # rotation + size change
        rot_sizeLayout = QHBoxLayout()
        rot_sizeLayout.addWidget(self.rotFrame)
        rot_sizeLayout.addWidget(self.sizeFrame)
        self.Layout.addLayout(rot_sizeLayout)
        self.Layout.addWidget(self.warningLBL)

        self.setLayout(self.Layout)

    def exec_(self):
        self.manage_rotation()
        self.manage_size()
        self.manage_mov()
        return super().exec_()

    def setRuler(self, rdy):
        if rdy:
            self.mmRDB.setEnabled(True)
            self.mRDB.setEnabled(True)
        else:
            self.mmRDB.setEnabled(False)
            self.mRDB.setEnabled(False)

    def manage_rotation(self):
        if len(self.rot_checkboxes) != 0:
            self.rotFrame.setHidden(False)
        else:
            self.rotFrame.setHidden(True)

    def manage_size(self):
        # print(len(self.size_checkboxes))
        if len(self.size_checkboxes) != 0:
            self.sizeFrame.setHidden(False)
        else:
            self.sizeFrame.setHidden(True)

    def manage_mov(self):
        if len(self.obj_checkboxes) != 0:
            self.objFrame.setHidden(False)
            self.warningLBL.setVisible(False)
        else:
            self.objFrame.setHidden(True)
            self.warningLBL.setVisible(True)

    def add_object(self, object_name):
        checkbox1 = QCheckBox(object_name)
        self.objNameLayout.addWidget(checkbox1)
        self.obj_checkboxes.append(checkbox1)
        checkbox2 = QCheckBox(object_name)
        self.sizeObjLayout.addWidget(checkbox2)
        self.size_checkboxes.append(checkbox2)

    def delete_object(self, object_name):
        if object_name == "ALL":
            for obj in self.obj_checkboxes:
                obj.deleteLater()
            for obj in self.size_checkboxes:
                obj.deleteLater()
            self.size_checkboxes.clear()
            self.obj_checkboxes.clear()
            return

        # delete from movement
        index = next(
            (
                i
                for i, item in enumerate(self.obj_checkboxes)
                if item.text() == object_name
            ),
            -1,
        )
        if index != -1:
            self.obj_checkboxes[index].deleteLater()
            del self.obj_checkboxes[index]

        # delet from size change
        index = next(
            (
                i
                for i, item in enumerate(self.size_checkboxes)
                if item.text() == object_name
            ),
            -1,
        )
        if index != -1:
            self.size_checkboxes[index].deleteLater()
            del self.size_checkboxes[index]

    def add_rotation(self, rot_name):
        checkbox1 = QCheckBox(rot_name)
        self.rotNameLayout.addWidget(checkbox1)
        self.rot_checkboxes.append(checkbox1)

    def delete_rotation(self, rot_name):
        if rot_name == "ALL":
            for rot in self.rot_checkboxes:
                rot.deleteLater()
            self.rot_checkboxes.clear()
            return

        # delete from movement
        index = next(
            (
                i
                for i, item in enumerate(self.rot_checkboxes)
                if item.text() == rot_name
            ),
            -1,
        )
        if index != -1:
            self.rot_checkboxes[index].deleteLater()
            del self.rot_checkboxes[index]

        # delet from size change
        index = next(
            (
                i
                for i, item in enumerate(self.rot_checkboxes)
                if item.text() == rot_name
            ),
            -1,
        )
        if index != -1:
            self.rot_checkboxes[index].deleteLater()
            del self.rot_checkboxes[index]

    def plot_movement(self):
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "MOV"
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()
        self.accept()

    def export_movement(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "MOV"
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()
        self.accept()

    def plot_rotation(self):
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "ROT"
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots
        self.get_rotation_parameters()
        self.accept()

    def export_rotation(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "ROT"
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots
        self.get_rotation_parameters()
        self.accept()

    def plot_size(self):
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "SIZ"
        objs = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.accept()

    def export_size(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "SIZ"
        objs = [
            checkbox.text() for checkbox in self.size_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.accept()

    def get_movement_parameters(self):
        # PROPERTY
        if self.posRDB.isChecked():
            self.parameters["prop"] = "POS"
        elif self.velRDB.isChecked():
            self.parameters["prop"] = "VEL"
        elif self.accRDB.isChecked():
            self.parameters["prop"] = "ACC"
        else:
            self.parameters.clear()
        # AXIS
        if self.xtRDB.isChecked():
            self.parameters["ax"] = "XT"
        elif self.ytRDB.isChecked():
            self.parameters["ax"] = "YT"
        else:
            self.parameters.clear()

        # UNITS
        if self.mmRDB.isChecked():
            self.parameters["unit"] = "mm"
        elif self.mRDB.isChecked():
            self.parameters["unit"] = "m"
        elif self.pixRDB.isChecked():
            self.parameters["unit"] = "pix"
        else:
            self.parameters.clear()

    def get_rotation_parameters(self):
        # print(self.parameters)
        # PROPERTY
        if self.rot_posRDB.isChecked():
            # print("pos is checked")
            self.parameters["prop"] = "POS"
            # print(self.parameters)
        elif self.rot_velRDB.isChecked():
            self.parameters["prop"] = "VEL"
            # print(self.parameters)
        elif self.rot_accRDB.isChecked():
            self.parameters["prop"] = "ACC"
            # print(self.parameters)
        else:
            self.parameters.clear()
            # print(self.parameters)

        # UNIT
        if self.degRDB.isChecked():
            self.parameters["unit"] = "DEG"
            # print(self.parameters)
        elif self.radRDB.isChecked():
            self.parameters["unit"] = "ROT"
            # print(self.parameters)
        else:
            self.parameters.clear()


class PlotDialog(QWidget):
    def __init__(self, df, title, unit, parent=None):
        super().__init__(parent)
        # self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Figure")
        self.setAttribute(Qt.WA_DeleteOnClose)
        # self.setModal(False)

        self.figure = Figure()

        ax = self.figure.add_subplot(111)
        ax.clear()

        df.plot(kind="line", x=0, ax=ax)
        ax.set_ylabel(unit)
        ax.set_title(title)

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
