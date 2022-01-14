from dis import Instruction
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
    QMessageBox
)
from PyQt5.QtGui import QMouseEvent, QCursor, QWheelEvent, QIntValidator
from PyQt5.QtCore import QLine, QThread, Qt, pyqtSignal, QEventLoop, QPoint, right
import math
import cv2
import numpy as np


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



class Rotation:
    def __init__(self, P1, P2):
        self.P1=P1
        self.P2=P2
        self.rotation=[]
    

    def calculate(self):
        P = np.vectorize(lambda P: P.point_path)
        
        path1=P(self.P1)
        path2=P(self.P2)
        rotation_init=np.arctan2(self.P2.point[1]-self.P1.point[1],self.P2.point[0]-self.P1.point[0])
        rot_array=-(np.arctan2(path2[:,1]-path1[:,1],path2[:,0]-path1[:,0])-rotation_init)/np.pi*180
        self.rotation=rot_array.tolist()



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




class TrackingThread(QThread):
    """QThread class responsible for running the tracking algorithm"""
    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success=pyqtSignal()
    rotation_calculated=pyqtSignal(Rotation)

    def __init__(
        self, objects_to_track, camera, start, stop, tracker_type, size, rotation_endpoints, fps
    ):
        self.objects_to_track = objects_to_track
        self.camera = camera
        self.section_start = start
        self.section_stop = stop
        self.tracker_type = tracker_type
        self.size = size
        self.rotation_endpoints = rotation_endpoints
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
                    # hibakezelés, nem jó a beolvasott frame, ez akár elő is fordulhat!
                    break

                # update the tracker
                ret, roi_box = tracker.update(frame)
                if ret:
                    # traditional tracking
                    M.rectangle_path.append(roi_box)
                    M.point_path.append(
                        (roi_box[0] + dx * roi_box[2], roi_box[1] + dy * roi_box[3])
                    )


                    # change of size
                    if self.size:
                        M.size_change.append((roi_box[2]/w0+roi_box[3]/h0)/2)
                    M.timestamp.append((i + 1) / self.fps)

                    #progress
                    self.progress = math.ceil(
                        i / (self.section_stop - self.section_start) * 100
                    )
                    self.progressChanged.emit(self.progress)
                    # self.status = f"Tracking ... {self.progress}"
                else:
                    #ha a frame jó de a tracker sikertelen akkor gond van, hibakezelés
                    pass
                    # self.status = "ERROR: Tracker update unsuccessful!"
                    # self.is_running = False

                if not self.is_running:
                    M.rectangle_path = []
                    M.timestamp = []
                    break

            if not self.is_running:
                break

            ### POST-PROCESSING START ###
            # HERE COMES THE FILTERS

        ### end of object-by-object for loop

        # rotation tracking
        if len(self.rotation_endpoints)>0:
            self.newObject.emit("Rotation")
            p1_index = next((i for i, item in enumerate(self.objects_to_track) if item.name == self.rotation_endpoints[0]), -1)
            p2_index = next((i for i, item in enumerate(self.objects_to_track) if item.name == self.rotation_endpoints[1]), -1)
            if p1_index!=-1 and p2_index!=-1:
                P1=self.objects_to_track[p1_index]
                P2=self.objects_to_track[p2_index]
                self.newObject.emit("Rotation between :"+P1.name+" and "+P2.name)
                R=Rotation(P1, P2)
                R.calculate()
                self.rotation_calculated.emit(R)

            else:
                #error, ennek nem szabadna megtörténnie
                pass

        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)
        if self.is_running:
            self.success.emit()




class VideoLabel(QLabel):
    """Label to display the frames from OpenCV"""
    press = pyqtSignal(float, float)
    moving = pyqtSignal(float, float)
    release = pyqtSignal(float, float)
    wheel=pyqtSignal(float)

    def __init__(self, parent=None):
        self.press_pos = None
        self.current_pos = None
        super(VideoLabel, self).__init__(parent)


    def wheelEvent(self, a0: QWheelEvent):
        if a0.angleDelta().y()>0:
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


class TrackingSettings(QDialog):
    """Modal dialog where users can specify the tracking settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking details")
        self.setModal(True)
        
        ### Additional Features ###
        self.rotationCHB=QCheckBox("Track rotation")
        self.rotationCHB.stateChanged.connect(self.openRotationSettings)
        self.sizeCHB=QCheckBox("Track size change")
        self.sizeCHB.stateChanged.connect(self.sizeMode)
        
        featureLayout=QVBoxLayout()
        featureLayout.addWidget(self.rotationCHB)
        featureLayout.addWidget(self.sizeCHB)

        featureGB=QGroupBox("Additional features to track")
        featureGB.setLayout(featureLayout)


        ### Tracking algorithm ###
        self.algoritmCMB = QComboBox()
        self.algoritmCMB.addItems(
            ["CSRT", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE"]
        )

        self.zoomNotificationLBL=QLabel("Only the CSRT algorithm is capable of handling the size change of an object!")
        self.zoomNotificationLBL.setVisible(False)
        algoLayout=QVBoxLayout()
        algoLayout.addWidget(self.algoritmCMB)
        algoLayout.addWidget(self.zoomNotificationLBL)

        algoGB=QGroupBox("Tracking algorithm")
        algoGB.setLayout(algoLayout)


        ### Left side veritcal layout ###
        leftLayout=QVBoxLayout()
        leftLayout.addWidget(featureGB)
        leftLayout.addWidget(algoGB)

        
        ### Real FPS input ###
        fpsLBL=QLabel("Real FPS:")
        self.fpsLNE=QLineEdit()
        self.fpsLNE.setValidator(QIntValidator(0, 1000000))

        fpsLayout=QHBoxLayout()
        fpsLayout.addWidget(fpsLBL)
        fpsLayout.addWidget(self.fpsLNE)

        ### Filter ###
        filterLBL=QLabel("Filter")
        self.filterCMB=QComboBox()
        
        filterHLayout=QHBoxLayout()
        filterHLayout.addWidget(filterLBL)
        filterHLayout.addWidget(self.filterCMB)

        filterBTN=QPushButton("Filter settings")
        filterBTN.clicked.connect(self.openFilterSettings)


        ### Derivative ###
        derivativeLBL=QLabel("Derivative:")
        self.derivativeCMB=QComboBox()


        derivativeHLayout=QHBoxLayout()
        derivativeHLayout.addWidget(derivativeLBL)
        derivativeHLayout.addWidget(self.derivativeCMB)

        derivativeBTN=QPushButton("Derivative settings")
        derivativeBTN.clicked.connect(self.openDerivativeSettings)


        ### Right side GroupBox ###
        rightLayout=QVBoxLayout()
        rightLayout.addLayout(fpsLayout)
        rightLayout.addLayout(filterHLayout)
        rightLayout.addWidget(filterBTN)
        rightLayout.addLayout(derivativeHLayout)
        rightLayout.addWidget(derivativeBTN)
        
        rightGB=QGroupBox("Post-processing settings")
        rightGB.setLayout(rightLayout)


        ### Settings horizontal layout ###
        settingsLayout=QHBoxLayout()
        settingsLayout.addLayout(leftLayout)
        settingsLayout.addWidget(rightGB)


        ### TRACK button and final layout
        trackBTN = QPushButton("Track")
        trackBTN.clicked.connect(self.accept)
        
        mainlayout=QVBoxLayout()
        mainlayout.addLayout(settingsLayout)
        mainlayout.addWidget(trackBTN)

        self.setLayout(mainlayout)


        ### Rotation settings dialog ####
        self.rotationSettings=RotationSettings()
        self.filterSettings=FilterSettings()
        self.derivativeSettings=DerivativeSettings()



    def openRotationSettings(self):
        if self.rotationCHB.isChecked():
            if self.rotationSettings.p1CMB.count()>=2:
                self.rotationSettings.exec_()
            else:
                self.rotationCHB.setCheckState(Qt.Unchecked)
                msg=QMessageBox()
                msg.setWindowTitle("Not enough points!")
                msg.setText("You need at least two points for rotation tracking!")
                msg.setIcon(QMessageBox.Warning)
                msg.exec_()
                self.rotationCHB.setChecked(False)
        


    def openFilterSettings(self):
        # collect data
        self.filterSettings.show()


    def openDerivativeSettings(self):
        # collect data
        self.derivativeSettings.show()


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

    
    def rotation(self):
        return self.rotationCHB



    def fps(self):
        return self.fpsLNE.text()


    
class RotationSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Rotation settings")
        self.setModal(True)
        instuctionLBL=QLabel("Select two points for for tracking")
        self.warningLBL=QLabel("You must select two different points!")
        self.warningLBL.setVisible(False)
        okBTN=QPushButton("Save")
        okBTN.clicked.connect(self.validate)
        self.p1CMB=QComboBox()
        self.p2CMB=QComboBox()
        layout=QVBoxLayout()
        layout.addWidget(instuctionLBL)
        layout.addWidget(self.p1CMB)
        layout.addWidget(self.p2CMB)
        layout.addWidget(okBTN)
        layout.addWidget(self.warningLBL)
        self.setLayout(layout)

    def validate(self):
        if self.p1CMB.currentText()!=self.p2CMB.currentText():
            self.accept()
        else:
            self.warningLBL.setVisible(True)
    

    def get_endpoints(self):
        return (self.p1CMB.currentText(), self.p2CMB.currentText())

        

class FilterSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Filter settings")
        self.setModal(True)


class DerivativeSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Derivative settings")
        self.setModal(True)



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