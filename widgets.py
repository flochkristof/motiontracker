from tkinter import EXCEPTION
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
from PyQt5.QtGui import (
    QMouseEvent,
    QCursor,
    QWheelEvent,
    QIntValidator,
    QIcon,
    QDoubleValidator,
)
from PyQt5.QtCore import QLine, QThread, Qt, pyqtSignal, QEventLoop, QPoint
import math
import cv2
import numpy as np
from funcs import *
from classes import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import pynumdiff

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
        for j in range(len(self.objects_to_track)):
            M = self.objects_to_track[j]

            # emit the name of the tracked object
            self.newObject.emit("Tracking object: " + M.name + "...")

            # reset previous data
            M.reset_data()
            if j == 0:
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
            # if self.tracker_type == "GOTURN":
            #    tracker = cv2.TrackerGOTURN_create()
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

            try:
                tracker.init(frame, rect2cropped(M.rectangle, self.roi_rect))
            except:
                self.is_running = False
                self.error_occured.emit("Unable to initialize thr tracker!")
                break

            M.rectangle_path.append(M.rectangle)
            M.point_path.append(M.point)
            if j == 0:
                self.timestamp.append(0)

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
                try:
                    ret, roi_box = tracker.update(frame)
                except Exception as e:
                    self.error_occured.emit(f"Tracking failed!\n{e}")
                    self.is_running = False
                    break

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
                        self.timestamp.append((i + 1) / self.fps)
                    # progress
                    self.progress = math.ceil(
                        i / (self.section_stop - self.section_start) * 100
                    )
                    self.progressChanged.emit(self.progress)
                    # self.status = f"Tracking ... {self.progress}"
                else:
                    self.error_occured.emit(
                        "Tracking failed!\n Tracker returned with failure!"
                    )
                    self.is_running = False

                if not self.is_running:
                    M.rectangle_path = []
                    break

            if not self.is_running:
                break
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)
        if self.is_running:
            self.success.emit()


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


class PostProcessSettings(QDialog):
    """Modal dialog for users to specify post processing settings"""

    diffdata = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Post-processing settings")
        self.setModal(True)
        self.setObjectName("postprocess_window")
        with open("style/postprocess.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon("images/logo.svg"))
        self.setMinimumWidth(600)

        # data output
        self.parameters = None

        # Create widgets
        diffFamilyLBL = QLabel("Differentiation algorithm family:")
        specificAlgoLBL = QLabel("Specified algorithm:")

        # Differentiation family
        self.diffFamilyCMB = QComboBox()
        self.diffFamilyCMB.setObjectName("diffFamilyCMB")

        self.diffFamilyCMB.addItems(
            [
                "Finite Difference",
                "Smooth Finite Difference",
                "Total Variation Regularization",
                "Linear Models",
            ]
        )
        self.diffFamilyCMB.currentTextChanged.connect(self.algoFamilyUpdated)

        # Finite difference for default
        self.diffSpecCMB = QComboBox()
        self.diffSpecCMB.setObjectName("diffSpecCMB")
        self.diffSpecCMB.addItems(
            [
                "First Order Finite Difference",
                "Iterated First Order Finite Difference",
                "Second Order Finite Difference",
            ]
        )
        self.diffSpecCMB.currentTextChanged.connect(self.diffSpecUpdated)

        diffFamLayout = QHBoxLayout()
        diffFamLayout.addWidget(diffFamilyLBL)
        diffFamLayout.addWidget(self.diffFamilyCMB)

        specDiffLayout = QHBoxLayout()
        specDiffLayout.addWidget(specificAlgoLBL)
        specDiffLayout.addWidget(self.diffSpecCMB)

        ## Possible input fields

        # Window size
        winSizeLBL = QLabel("Window Size:")
        self.winSizeLNE = QLineEdit()
        self.winSizeLNE.setValidator(QIntValidator(0, 1000))

        winSizeLayout = QHBoxLayout()
        winSizeLayout.addWidget(winSizeLBL)
        winSizeLayout.addWidget(self.winSizeLNE)

        self.winSizeGB = QGroupBox("")
        self.winSizeGB.setStyleSheet("border:0;")
        self.winSizeGB.setLayout(winSizeLayout)

        # Order
        orderLBL = QLabel("Order:")
        self.orderLNE = QLineEdit()
        self.orderLNE.setValidator(QIntValidator(0, 30))

        orderLayout = QHBoxLayout()
        orderLayout.addWidget(orderLBL)
        orderLayout.addWidget(self.orderLNE)

        self.orderGB = QGroupBox("")
        self.orderGB.setStyleSheet("border:0;")
        self.orderGB.setLayout(orderLayout)

        # Iterations
        iterationsLBL = QLabel("Iterations:")
        self.iterationsLNE = QLineEdit()
        self.iterationsLNE.setValidator(QIntValidator(0, 50))
        self.iterationsLNE.setFixedWidth(100)

        iterationsLayout = QHBoxLayout()
        iterationsLayout.addWidget(iterationsLBL)
        iterationsLayout.addWidget(self.iterationsLNE)

        self.iterationsGB = QGroupBox("")
        self.iterationsGB.setStyleSheet("border:0;")
        self.iterationsGB.setLayout(iterationsLayout)

        # Cutoff frequency
        cutoffLBL = QLabel("Cutoff Frequency:")
        self.cutoffLNE = QLineEdit()
        self.cutoffLNE.setValidator(QDoubleValidator())

        cutoffLayout = QHBoxLayout()
        cutoffLayout.addWidget(cutoffLBL)
        cutoffLayout.addWidget(self.cutoffLNE)

        self.cutoffGB = QGroupBox("")
        self.cutoffGB.setStyleSheet("border:0;")
        self.cutoffGB.setLayout(cutoffLayout)

        # Smoothing factor
        smoothFactLBL = QLabel("Smoothing factor:")
        self.smoothFactLNE = QLineEdit()
        self.smoothFactLNE.setValidator(QIntValidator(0, 50))

        smoothFactLayout = QHBoxLayout()
        smoothFactLayout.addWidget(smoothFactLBL)
        smoothFactLayout.addWidget(self.smoothFactLNE)

        self.smoothFactGB = QGroupBox("")
        self.smoothFactGB.setStyleSheet("border:0;")
        self.smoothFactGB.setLayout(smoothFactLayout)

        # Regularization parameter
        regParLBL = QLabel("Regularization parameter:")
        self.regParLNE = QLineEdit()
        self.regParLNE.setValidator(QDoubleValidator())

        regParLayout = QHBoxLayout()
        regParLayout.addWidget(regParLBL)
        regParLayout.addWidget(self.regParLNE)

        self.regParGB = QGroupBox("")
        self.regParGB.setStyleSheet("border:0;")
        self.regParGB.setLayout(regParLayout)

        # Step size
        stepLBL = QLabel("Step Size:")
        self.stepLNE = QLineEdit()
        self.stepLNE.setValidator(QIntValidator(1, 10000))

        stepLayout = QHBoxLayout()
        stepLayout.addWidget(stepLBL)
        stepLayout.addWidget(self.stepLNE)

        self.stepGB = QGroupBox("")
        self.stepGB.setStyleSheet("border:0;")
        self.stepGB.setLayout(stepLayout)

        # Kernel
        kernelLBL = QLabel("Kernel:")
        self.kernelCMB = QComboBox()
        self.kernelCMB.addItems(["Gaussian", "Friedrichs"])
        kernelLayout = QHBoxLayout()
        kernelLayout.addWidget(kernelLBL)
        kernelLayout.addWidget(self.kernelCMB)

        self.kernelGB = QGroupBox("")
        self.kernelGB.setStyleSheet("border:0;")
        self.kernelGB.setLayout(kernelLayout)

        # Sliding
        slidingLBL = QLabel("Use sliding:")
        self.slidingCMB = QComboBox()
        self.slidingCMB.addItems(["Yes", "No"])
        slidingLayout = QHBoxLayout()
        slidingLayout.addWidget(slidingLBL)
        slidingLayout.addWidget(self.slidingCMB)

        self.slidingGB = QGroupBox("")
        self.slidingGB.setStyleSheet("border:0;")
        self.slidingGB.setLayout(slidingLayout)

        # No input alert
        self.noInputSpecLBL = QLabel("No parameters to configure")
        self.noInputSpecLBL.setObjectName("noInputLBL")

        self.manSpecRDB = QRadioButton("Manually Specified")
        self.manSpecRDB.toggled.connect(self.modeUpdated)

        # Paramters into Layout and list for control
        manSpecLayout = QVBoxLayout()
        manSpecLayout.addWidget(self.manSpecRDB)
        manSpecLayout.addWidget(self.noInputSpecLBL)
        manSpecLayout.addWidget(self.winSizeGB)
        manSpecLayout.addWidget(self.orderGB)
        manSpecLayout.addWidget(self.iterationsGB)
        manSpecLayout.addWidget(self.cutoffGB)
        manSpecLayout.addWidget(self.smoothFactGB)
        manSpecLayout.addWidget(self.regParGB)
        manSpecLayout.addWidget(self.stepGB)
        manSpecLayout.addWidget(self.slidingGB)
        manSpecLayout.addWidget(self.kernelGB)

        self.parameters_list = [
            self.winSizeGB,
            self.orderGB,
            self.iterationsGB,
            self.cutoffGB,
            self.smoothFactGB,
            self.regParGB,
            self.stepGB,
            self.slidingGB,
            self.kernelGB,
        ]
        for p in self.parameters_list:
            p.setVisible(False)
            p.setFixedWidth(300)
            p.setEnabled(False)

        # Optimization based input parameters
        self.optRDB = QRadioButton("Optimization based")
        self.optRDB.toggled.connect(self.modeUpdated)

        cutoffOptLBL = QLabel("Cutoff Frequency:")
        self.cutoffOptLNE = QLineEdit()
        self.cutoffOptLNE.setValidator(QDoubleValidator())

        cutoffOptLayout = QHBoxLayout()
        cutoffOptLayout.addWidget(cutoffOptLBL)
        cutoffOptLayout.addWidget(self.cutoffOptLNE)

        self.cutoffOptGB = QGroupBox("")
        self.cutoffOptGB.setStyleSheet("border:0;")
        self.cutoffOptGB.setLayout(cutoffOptLayout)
        self.cutoffOptGB.setVisible(False)

        optLayout = QVBoxLayout()
        optLayout.setAlignment(Qt.AlignTop)
        optLayout.addWidget(self.optRDB)
        optLayout.addWidget(self.cutoffOptGB)

        parametersLayout = QHBoxLayout()
        parametersLayout.setAlignment(Qt.AlignTop)
        parametersLayout.addLayout(manSpecLayout)
        parametersLayout.addLayout(optLayout)

        # Parameters groupbox
        parametersGB = QGroupBox("Parameters")
        parametersGB.setLayout(parametersLayout)

        calculateBTN = QPushButton("Calculate")
        calculateBTN.setObjectName("calculateBTN")
        calculateBTN.clicked.connect(self.accept)

        # Main layout
        Layout = QVBoxLayout()
        Layout.addLayout(diffFamLayout)
        Layout.addLayout(specDiffLayout)
        Layout.addWidget(parametersGB)
        Layout.addWidget(calculateBTN)
        self.setLayout(Layout)

    def algoFamilyUpdated(self):
        """Update the specific fillter combobox according to family"""
        if self.diffFamilyCMB.currentText() == "Finite Difference":
            self.diffSpecCMB.clear()
            self.diffSpecCMB.addItems(
                [
                    "First Order Finite Difference",
                    "Iterated First Order Finite Difference",
                    "Second Order Finite Difference",
                ]
            )
        elif self.diffFamilyCMB.currentText() == "Smooth Finite Difference":
            self.diffSpecCMB.clear()
            self.diffSpecCMB.addItems(
                [
                    "Finite Difference with Median Smoothing",
                    "Finite Difference with Mean Smoothing",
                    "Finite Difference with Gaussian Smoothing",
                    "Finite Difference with Friedrichs Smoothing",
                    "Finite Difference with Butterworth Smoothing",
                    "Finite Difference with Spline Smoothing",
                ]
            )
        elif self.diffFamilyCMB.currentText() == "Total Variation Regularization":
            self.diffSpecCMB.clear()
            self.diffSpecCMB.addItems(
                [
                    "Iterative Total Variation Regularization with Regularized Velocity",
                    "Convex Total Variation Regularization With Regularized Velocity",
                    "Convex Total Variation Regularization with Regularized Acceleration",
                    "Convex Total Variation Regularization with Regularized Jerk",
                    "Convex Total Variation Regularization with Sliding Jerk",
                    "Convex Total Variation Regularization with Smoothed Acceleration",
                ]
            )
        elif self.diffFamilyCMB.currentText() == "Linear Models":
            self.diffSpecCMB.clear()
            self.diffSpecCMB.addItems(
                [
                    "Spectral Derivative",
                    "Sliding Polynomial Derivative",
                    "Savitzky-Golay Filter",
                    "Sliding Chebychev Polynomial Fit",
                ]
            )

    def modeUpdated(self):
        if self.manSpecRDB.isChecked():
            for p in self.parameters_list:
                p.setEnabled(True)
            self.cutoffOptGB.setEnabled(False)
        elif self.optRDB.isChecked():
            self.cutoffOptGB.setEnabled(True)
            for p in self.parameters_list:
                p.setEnabled(False)

    def diffSpecUpdated(self):
        algo = self.diffSpecCMB.currentText()
        # ["First Order Finite Difference", "Iterated First Order Fintie Difference", "Second Order Finite Difference"]
        if algo == "First Order Finite Difference":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)

            # custom
            self.cutoffOptGB.setVisible(False)
            self.noInputSpecLBL.setVisible(True)
        elif algo == "Iterated First Order Finite Difference":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.iterationsGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)

        elif algo == "Second Order Finite Difference":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)

            # custom
            self.cutoffOptGB.setVisible(False)
            self.noInputSpecLBL.setVisible(True)

        # ["Finite Difference with Median Smoothing","Finite Difference with Mean Smoothing","Finite Difference with Gaussian Smoothing","Finite Difference with Friedrichs Smoothing","Finite Difference with Butterworth Smoothing","Finite Difference with Spline Smoothing",
        elif algo in {
            "Finite Difference with Median Smoothing",
            "Finite Difference with Mean Smoothing",
            "Finite Difference with Gaussian Smoothing",
            "Finite Difference with Friedrichs Smoothing",
        }:
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.noInputSpecLBL.setVisible(False)
            self.iterationsGB.setVisible(True)
            self.winSizeGB.setVisible(True)
        elif algo == "Finite Difference with Butterworth Smoothing":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.noInputSpecLBL.setVisible(False)
            self.orderGB.setVisible(True)
            self.cutoffGB.setVisible(True)
        elif algo == "Finite Difference with Spline Smoothing":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.orderGB.setVisible(True)
            self.smoothFactGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)
        # ["Iterative Total Variation Regularization with Regularized Velocity","Convex Total Variation Regularization With Regularized Velocity","Convex Total Variation Regularization with Regularized Acceleration","Convex Total Variation Regularization with Regularized Jerk","Convex Total Variation Regularization with Sliding","Convex Total Variation Regularization with Smoothed Acceleration",
        elif algo in {
            "Iterative Total Variation Regularization with Regularized Velocity",
            "Convex Total Variation Regularization with Sliding Jerk",
        }:
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.iterationsGB.setVisible(True)
            self.regParGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)
        elif algo in {
            "Convex Total Variation Regularization With Regularized Velocity",
            "Convex Total Variation Regularization with Regularized Acceleration",
            "Convex Total Variation Regularization with Regularized Jerk",
        }:
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.regParGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)
        elif algo == "Convex Total Variation Regularization with Smoothed Acceleration":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.regParGB.setVisible(True)
            self.winSizeGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)
        # "Spectral Derivative","Sliding Polynomial Derivative","Savitzky-Golay Filter","Sliding Chebychev Polynomial Fit",

        elif algo in {
            "Sliding Polynomial Derivative",
            "Sliding Chebychev Polynomial Fit",
        }:
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.orderGB.setVisible(True)
            self.winSizeGB.setVisible(True)
            self.slidingGB.setVisible(True)
            self.stepGB.setVisible(True)
            self.kernelGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)
        elif algo == "Savitzky-Golay Filter":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.orderGB.setVisible(True)
            self.winSizeGB.setVisible(True)

        elif algo == "Spectral Derivative":
            # reset
            for p in self.parameters_list:
                p.setVisible(False)
            self.cutoffOptGB.setVisible(True)

            # cutom
            self.cutoffGB.setVisible(True)
            self.noInputSpecLBL.setVisible(False)

    def collectParameters(self):
        algo = self.diffSpecCMB.currentText()
        # differenciálás paraméterek nélkül
        if algo in {"First Order Finite Difference", "Second Order Finite Difference"}:
            return (False, algo, None)
        # Optimizáció alapú paraméterkeresés
        elif self.optRDB.isChecked():
            if self.cutoffOptLNE.text() != "":
                return (True, algo, float(self.cutoffOptLNE.text()))
            else:
                return None
        elif self.manSpecRDB.isChecked():
            if algo == "Iterated First Order Finite Difference":
                if self.iterationsLNE.text() != "":
                    return (
                        False,
                        algo,
                        [int(self.iterationsLNE.text())],
                        {"iterate": True},
                    )
                else:
                    return None
            elif algo in {
                "Finite Difference with Median Smoothing",
                "Finite Difference with Mean Smoothing",
                "Finite Difference with Gaussian Smoothing",
                "Finite Difference with Friedrichs Smoothing",
            }:
                if (self.iterationsLNE.text != "") and (self.winSizeLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [int(self.winSizeLNE.text()), int(self.iterationsLNE.text())],
                        {},
                    )
                else:
                    return None
            elif algo == "Finite Difference with Butterworth Smoothing":
                if (self.orderLNE.text() != "") and (self.cutoffLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [int(self.orderLNE.text()), float(self.cutoffLNE.text()),],
                        {
                            "padmethod": "gust"
                        },  # TODO: needs to be normalized before calculations
                    )
                else:
                    return None
            elif algo == "Finite Difference with Spline Smoothing":
                if (self.orderLNE.text() != "") and (self.smoothFactLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [int(self.orderLNE.text()), float(self.smoothFactLNE.text())],
                        {},
                    )
                else:
                    return None
            elif algo == "Convex Total Variation Regularization with Sliding Jerk":
                if (self.iterationsLNE.text() != "") and (self.regParLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [float(self.regParLNE.text()), int(self.iterationsLNE.text())],
                        {"solver": "CVXOPT", "iterate": True},
                    )
                else:
                    return None
            elif (
                algo
                == "Iterative Total Variation Regularization with Regularized Velocity"
            ):
                if (self.iterationsLNE.text() != "") and (self.regParLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [int(self.regParLNE.text()), int(self.iterationsLNE.text())],
                        {"solver": "CVXOPT", "iterate": True},
                    )
                else:
                    return None
            elif algo in {
                "Convex Total Variation Regularization with Regularized Velocity",
                "Convex Total Variation Regularization with Regularized Acceleration",
                "Convex Total Variation Regularization with Regularized Jerk",
            }:
                if self.regParLNE.text != "":
                    return (
                        False,
                        algo,
                        [float(self.regParLNE.text())],
                        {"solver": "CVXOPT"},
                    )
                else:
                    return None
            elif (
                algo
                == "Convex Total Variation Regularization with Smoothed Acceleration"
            ):
                if (self.regParLNE.text() != "") and (self.winSizeLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [float(self.regParLNE.text()), int(self.winSizeLNE.text())],
                        {"solver": "CVXOPT", "iterate": True},
                    )
                else:
                    return None
            elif algo in {
                "Sliding Polynomial Derivative",
                "Sliding Chebychev Polynomial Fit",
            }:
                if (self.orderLNE.text() != "") and (self.winSizeLNE.text() != ""):
                    if self.slidingCMB.currentText() == "Yes":
                        if self.stepLNE.text() != "":
                            return (
                                False,
                                algo,
                                [
                                    int(self.orderLNE.text()),
                                    int(self.winSizeLNE.text()),
                                ],
                                {
                                    "sliding": True,
                                    "step_size": int(self.stepLNE.text()),
                                    "kernel_name": self.kernelCMB.currentText().lower(),
                                },
                            )
                        else:
                            return (
                                False,
                                algo,
                                [
                                    int(self.orderLNE.text()),
                                    int(self.winSizeLNE.text()),
                                ],
                                {
                                    "sliding": False,
                                    "kernel_name": self.kernelCMB.currentText().lower(),
                                },
                            )

                    return (
                        False,
                        algo,
                        [int(self.orderLNE.text()), int(self.winSizeLNE.text())],
                        {},
                    )
                else:
                    return None
            elif algo == "Savitzky-Golay Filter":
                if (self.orderLNE.text() != "") and (self.winSizeLNE.text() != ""):
                    return (
                        False,
                        algo,
                        [
                            int(self.orderLNE.text()),
                            int(self.winSizeLNE.text()),
                            int(self.winSizeLNE.text()),
                        ],
                        {},
                    )
                else:
                    return None

            elif algo == "Spectral Derivative":
                if self.cutoffLNE.text() != "":
                    return (
                        False,
                        algo,
                        [float(self.cutoffLNE.text())],
                        {"even_extension": False, "pad_to_zero_dxdt": False},
                    )
        return None

    def accept(self):
        self.parameters = self.collectParameters()
        if self.parameters is None:
            return
        super().accept()


class TrackingSettings(QDialog):
    """Modal dialog for users to specify the tracking settings"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking settings")
        self.setModal(True)
        with open("style/tracking.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon("images/logo.svg"))
        self.setObjectName("tracker_window")
        self.setFixedSize(270, 200)

        # Creating the widgets
        algoLBL = QLabel("Tracking Algorithm:")
        self.algoCMB = QComboBox()
        self.algoCMB.addItems(
            ["CSRT", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE"]
        )
        sizeLBL = QLabel("Track the size change of objects:")
        self.sizeCHB = QCheckBox()
        self.sizeCHB.setLayoutDirection(Qt.RightToLeft)
        self.sizeCHB.stateChanged.connect(self.sizeMode)

        fpsLBL = QLabel("Real FPS:")
        self.fpsLNE = QLineEdit()
        self.fpsLNE.setValidator(QIntValidator(1, 100000))
        self.notificationLBL = QLabel(
            "Important information: Only the CSRF algorithm is capable of tracking the size change of an object!"
        )
        self.notificationLBL.setVisible(False)
        self.notificationLBL.setWordWrap(True)

        trackBTN = QPushButton("Track")
        trackBTN.clicked.connect(self.accept)
        trackBTN.setObjectName("trackBTN")

        # Organizing widgets into layouts
        algoLayout = QHBoxLayout()
        algoLayout.addWidget(algoLBL)
        algoLayout.addWidget(self.algoCMB)

        sizeLayout = QHBoxLayout()
        sizeLayout.addWidget(sizeLBL)
        sizeLayout.addWidget(self.sizeCHB)

        fpsLayout = QHBoxLayout()
        fpsLayout.addWidget(fpsLBL)
        fpsLayout.addWidget(self.fpsLNE)

        Layout = QVBoxLayout()
        Layout.addLayout(algoLayout)
        Layout.addLayout(sizeLayout)
        Layout.addLayout(fpsLayout)
        Layout.addWidget(self.notificationLBL)
        Layout.addItem(QSpacerItem(0, 60, QSizePolicy.Maximum, QSizePolicy.Expanding))
        Layout.addWidget(trackBTN)
        self.setLayout(Layout)

    def sizeMode(self):
        if self.sizeCHB.isChecked():
            self.notificationLBL.setVisible(True)
            self.algoCMB.setCurrentText("CSRT")
            self.algoCMB.setEditable(False)
            self.algoCMB.setEnabled(False)
        else:
            self.notificationLBL.setVisible(False)
            self.algoCMB.setEnabled(True)


"""class TrackingSettings(QDialog):
    Modal dialog where users can specify the tracking settings

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking details")
        self.setModal(True)
        with open("style/tracking.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon("images/logo.svg"))
        self.setObjectName("tracker_window")

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
            "Important! Only the CSRT algorithm is capable of handling the size change of an object!"
        )
        self.zoomNotificationLBL.setVisible(False)
        self.zoomNotificationLBL.setWordWrap(True)
        self.zoomNotificationLBL.setObjectName("zoomnotificationLBL")

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
        filterLBL = QLabel("Filter:")
        self.filterCMB = QComboBox()
        self.filterCMB.addItem("None")
        self.filterCMB.addItem("Gaussian")
        self.filterCMB.addItem("Moving AVG")
        self.filterCMB.addItem("Savitzky-Golay")
        self.filterCMB.currentTextChanged.connect(self.openFilterSettings)
        self.filterCMB.setObjectName("filterCMB")

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
        self.derivativeCMB.setObjectName("derivativeCMB")

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
        trackBTN.setObjectName("trackBTN")

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
"""


class RotationSettings(QDialog):
    """Dialog to select points for rotation tracking"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Rotation settings")
        self.setWindowIcon(QIcon("images/logo.svg"))
        self.setObjectName("rotation")
        with open("style/rotation.qss", "r") as style:
            self.setStyleSheet(style.read())
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
        self.setWindowIcon(QIcon("images/logo.svg"))
        with open("style/filter-derivative.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setObjectName("filter")
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
        self.setWindowIcon(QIcon("images/logo.svg"))
        with open("style/filter-derivative.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setObjectName("derivative")
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

        # FINITE DIFFERENCES
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
        self.setWindowIcon(QIcon("images/logo.svg"))
        with open("style/progress.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setObjectName("progress")
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


class PostProcesserThread(QThread):
    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success = pyqtSignal()
    error_occured = pyqtSignal(str)

    def __init__(self, mode, objects_to_track, dt, parameters):
        self.objects_to_track = objects_to_track
        self.dt = dt
        self.parameters = parameters
        self.progress = 0
        self.is_running = True
        self.mode = mode
        super(PostProcesserThread, self).__init__()

    def cancel(self):
        self.is_running = False

    def run(self):
        if self.mode:
            if self.parameters is None:
                self.error_occured.emit("Error: Invalid parameters!")
                return

            for i in range(len(self.objects_to_track)):
                M = self.objects_to_track[i]

                M.reset_output()

                x = np.array([p[0] for p in M.point_path])
                y = np.array([p[1] for p in M.point_path])

                # X coordinate
                if self.parameters[0]:
                    ret, xs, vx, ax = optimize_and_differentiate(
                        x, self.dt, self.parameters
                    )
                else:
                    ret, xs, vx, ax = differentiate(x, self.dt, self.parameters)

                if not ret:
                    self.error_occured.emit(
                        "Error: A porblem occured while calculating the derivative!"
                    )
                    self.is_running = False
                    break
                self.progressChanged.emit(
                    int(100 / (2 * len(self.objects_to_track)) * (i + 1))
                )

                # Y coordinate
                if self.parameters[0]:
                    ret, ys, vy, ay = optimize_and_differentiate(
                        y, self.dt, self.parameters
                    )
                else:
                    ret, ys, vy, ay = differentiate(y, self.dt, self.parameters)

                if not ret:
                    self.error_occured.emit(
                        "Error: A porblem occupred while calculating the derivative!"
                    )
                    self.is_running = False
                    break
                self.progressChanged.emit(
                    int(100 / len(self.objects_to_track) * (i + 1))
                )

                # Smoothed postion
                M.position = np.zeros((len(xs), 2))
                M.position[:, 0] = xs
                M.position[:, 1] = ys

                # Smoothed postion
                M.velocity = np.zeros((len(vx), 2))
                M.velocity[:, 0] = vx
                M.velocity[:, 1] = vy

                # Smoothed postion
                M.acceleration = np.zeros((len(ax), 2))
                M.acceleration[:, 0] = ax
                M.acceleration[:, 1] = ay
            self.success.emit()
        else:
            # here objetcts to track means only a single rotation object
            r = self.objects_to_track.rotation
            if self.parameters[0]:
                ret, rs, ang_v, ang_a = optimize_and_differentiate(
                    r, self.dt, self.parameters
                )
            else:
                ret, rs, ang_v, ang_a = differentiate(r, self.dt, self.parameters)
            if not ret:
                self.error_occured.emit(
                    "Error: A porblem occured while calculating the derivative!"
                )
                self.is_running = False
                return

            # Smoothed postion
            self.objects_to_track.rotation = rs
            self.objects_to_track.ang_velocity = ang_v
            self.objects_to_track.ang_acceleration = ang_a
        self.success.emit()


class ExportDialog(QDialog):
    """Handles the exporting of the data collected by the trackers"""

    export = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowIcon(QIcon("images/logo.svg"))
        self.setObjectName("export_dialog")
        with open("style/export.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowTitle("Export options")
        self.setModal(False)

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
        self.bothRDB = QRadioButton("Both")

        axBGP = QGroupBox("Axis")
        axLayout = QVBoxLayout()
        axLayout.addWidget(self.xtRDB)
        axLayout.addWidget(self.ytRDB)
        axLayout.addWidget(self.bothRDB)
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

    def show(self, size, rot, mov):
        self.manage_rotation(rot)
        self.manage_size(size)
        self.manage_mov(mov)
        return super().show()

    def setRuler(self, rdy):
        if rdy:
            self.mmRDB.setEnabled(True)
            self.mRDB.setEnabled(True)
        else:
            self.mmRDB.setEnabled(False)
            self.mRDB.setEnabled(False)

    def manage_rotation(self, rot):
        if len(self.rot_checkboxes) != 0:
            self.rotFrame.setHidden(False)
            if rot:
                self.rot_velRDB.setEnabled(True)
                self.rot_accRDB.setEnabled(True)
            else:
                self.rot_velRDB.setEnabled(False)
                self.rot_accRDB.setEnabled(False)
        else:
            self.rotFrame.setHidden(True)

    def manage_size(self, size):
        if size != 0:
            self.sizeFrame.setHidden(False)
        else:
            self.sizeFrame.setHidden(True)

    def manage_mov(self, mov):
        if len(self.obj_checkboxes) != 0:
            self.objFrame.setHidden(False)
            self.warningLBL.setVisible(False)
            if mov:
                self.accRDB.setVisible(True)
                self.velRDB.setVisible(True)
            else:
                self.accRDB.setVisible(False)
                self.velRDB.setVisible(False)
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
        print("plot movement function called!")
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "MOV"
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()
        if (
            "mode"
            and "units"
            and "out"
            and "objects"
            and "prop"
            and "ax" in self.parameters
        ):
            print("plot signal emitted")
            self.export.emit(self.parameters)
            # self.accept()

    def export_movement(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "MOV"
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()
        if (
            "mode"
            and "units"
            and "out"
            and "objects"
            and "prop"
            and "ax" in self.parameters
        ):
            self.export.emit(self.parameters)
            print("plot signal emitted")
            # self.accept()

    def plot_rotation(self):
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "ROT"
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots
        self.get_rotation_parameters()
        if "mode" and "units" and "out" and "rotations" and "prop" in self.parameters:
            self.export.emit(self.parameters)
            print("plot signal emitted")
            # self.accept()

    def export_rotation(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "ROT"
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots
        self.get_rotation_parameters()
        if "mode" and "units" and "out" and "rotations" and "prop" in self.parameters:
            self.export.emit(self.parameters)
            print("plot signal emitted")
            # self.accept()

    def plot_size(self):
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "SIZ"
        objs = [
            checkbox.text() for checkbox in self.size_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        print("plot signal emitted")
        self.export.emit(self.parameters)
        # self.accept()

    def export_size(self):
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "SIZ"
        objs = [
            checkbox.text() for checkbox in self.size_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        print("plot signal emitted")
        self.export.emit(self.parameters)
        # self.accept()

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
        elif self.bothRDB.isChecked():
            self.parameters["ax"] = "BOTH"
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
            self.parameters["unit"] = "RAD"
            # print(self.parameters)
        else:
            self.parameters.clear()


class PlotDialog(QWidget):
    def __init__(self, df, title, unit, parent=None):
        super().__init__(parent)
        # self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Figure")
        self.setWindowIcon(QIcon("images/logo.svg"))
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

