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
)
from PyQt5.QtGui import QMouseEvent, QCursor
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QEventLoop, QPoint
import math
import cv2


class TrackingThread(QThread):
    """QThread class responsible for running the tracking algorithm"""
    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)

    def __init__(
        self, objects_to_track, camera, start, stop, tracker_type, zoom, rotation, fps
    ):
        self.objects_to_track = objects_to_track
        self.camera = camera
        self.section_start = start
        self.section_stop = stop
        self.tracker_type = tracker_type
        self.zoom = zoom
        self.rotation = rotation
        self.fps = fps
        self.progress = "0"
        self.is_running = True
        super(TrackingThread, self).__init__()

    def cancel(self):
        self.is_running = False

    def run(self):
        for M in self.objects_to_track:
            # emit the name of the tracked object
            self.newObject.emit(M.name)

            # reset previous data
            M.timestamp = []
            M.rectangle_path = []  
            M.point_path = []
            M.size_change = []

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
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

            # initialize tracker
            ret, frame = self.camera.read()
            if not ret:
                # hibakezelés
                return
            tracker.init(frame, M.rectangle)

            # for the calculation of the point
            dy = (M.point[1] - M.rectangle[1]) / M.rectangle[3]
            dx = (M.point[0] - M.rectangle[0]) / M.rectangle[2]

            # for zoom
            w0=M.rectangle[2]
            h0=M.rectangle[3]

            # tracking
            for i in range(int(self.section_stop - self.section_start)):
                # read the next frame
                ret, frame = self.camera.read()
                if not ret:
                    # hibakezelés
                    break

                # update the tracker
                ret, roi_box = tracker.update(frame)
                if ret:

                    M.rectangle_path.append(roi_box)
                    M.point_path.append(
                        (roi_box[0] + dx * roi_box[2], roi_box[1] + dy * roi_box[3])
                    )
                    M.timestamp.append((i + 1) / self.fps)
                    #M.size_change.append((roi_box[2]/w0+roi_box[3]/h0)/2)
                    #print(f"{M.size_change[-1]}")
                    self.progress = math.ceil(
                        i / (self.section_stop - self.section_start) * 100
                    )
                    self.progressChanged.emit(self.progress)
                    # self.status = f"Tracking ... {self.progress}"
                else:
                    pass
                    # self.status = "ERROR: Tracker update unsuccessful!"
                    # self.is_running = False

                if not self.is_running:
                    M.rectangle_path = []
                    M.timestamp = []
                    break

            if not self.is_running:
                break
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)


class Motion:
    """Class that stores every detail of the objects being tracked"""
    def __init__(self, name, point, rectangle, visible=True):
        # identification
        self.name = name

        # starting properties
        self.point = point
        self.rectangle = rectangle
        self.visible = visible

        # output
        self.timestamp = []
        self.rectangle_path = []
        self.point_path = []
        self.size_change=[]


class Ruler:
    """Class for users to define a milimeter scale on the video frames"""
    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.mm = None
        self.mm_per_pix = None
        self.rdy = False
        self.visible = True

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
    """Label to display the frames from OpenCV"""
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


class TrackingSettings(QDialog):
    """Modal dialog where users can specify the tracking settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking details")
        trackLBL = QLabel("Tracking algorithm:")
        self.algoritmCMB = QComboBox()
        self.algoritmCMB.addItems(
            ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]
        )
        algoLayout = QHBoxLayout()
        algoLayout.addWidget(trackLBL)
        algoLayout.addWidget(self.algoritmCMB)

        propLayout = QHBoxLayout()
        propLBL = QLabel("Properties to track:")
        propLabelLayout = QVBoxLayout()
        propLabelLayout.addWidget(propLBL)
        propLabelLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        propCheckLayout = QVBoxLayout()
        self.XYcoordinatesCHB = QCheckBox("X and Y coordinates")
        self.rotationCHB = QCheckBox("Rotation around Z")
        self.zoomCHB = QCheckBox("Zoom")
        propCheckLayout.addWidget(self.XYcoordinatesCHB)
        propCheckLayout.addWidget(self.rotationCHB)
        propCheckLayout.addWidget(self.zoomCHB)

        propLayout.addLayout(propLabelLayout)
        propLayout.addLayout(propCheckLayout)

        trackLayout = QVBoxLayout()
        trackLayout.addLayout(algoLayout)
        trackLayout.addLayout(propLayout)

        trackGB = QGroupBox()
        trackGB.setTitle("Tracking settings")
        trackGB.setLayout(trackLayout)

        postprocessGB = QGroupBox()
        postprocessGB.setTitle("Post-processing settings")

        groupBoxLayout = QHBoxLayout()
        groupBoxLayout.addWidget(trackGB)
        groupBoxLayout.addWidget(postprocessGB)

        dialogLayout = QVBoxLayout()
        dialogLayout.addLayout(groupBoxLayout)

        startBTN = QPushButton()
        startBTN.setText("Start")
        startBTN.clicked.connect(self.accept)
        dialogLayout.addWidget(startBTN)

        self.setLayout(dialogLayout)

    def tracker_type(self):
        return self.algoritmCMB.currentText()

    def zoom(self):
        return self.zoomCHB.isChecked()

    def xy(self):
        return self.XYcoordinatesCHB.isChecked()

    def rotation(self):
        return self.rotationCHB.isChecked()


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
        self.label.setText("Tracking object: " + name)

    def updateBar(self, value):
        self.progressbar.setValue(value)


class ExportDialog(QDialog):
    """Handles the exporting of the data collected by the trackers"""
    def __init__(self, parent=None):
        super().__init__(parent)