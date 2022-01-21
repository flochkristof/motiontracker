import sys
from turtle import position
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpacerItem,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
    QGroupBox,
    QMessageBox,
)
import cv2
from numpy import rad2deg
from widgets import (
    ExportDialog,
    ExportingThread,
    Motion,
    PlotDialog,
    TrackingSettings,
    TrackingThread,
    VideoLabel,
    ObjectListWidget,
    Ruler,
    TrackingProgress,
)  # TimeInputDialog
from math import floor, ceil
from funcs import *
from classes import *
import pandas as pd


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()

        self.camera = None
        self.fps = None
        self.num_of_frames = 0
        self.x_offset = 0
        self.y_offset = 0
        self.zoom = 1  # from 1 goes down to 0
        self.section_start = None
        self.section_stop = None
        self.roi_rect = None
        self.filename = ""
        self.video_width = None
        self.video_height = None
        self.objects_to_track = []
        self.rotations = []
        self.timestamp = []
        self.point_tmp = None
        self.rect_tmp = None  # x0, y0, x1, y1
        self.ruler = Ruler()
        self.mode = False  # False-> before tracking; True->after tracking

        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.setWindowTitle("VideoWidget")
        self.setGeometry(100, 100, 1280, 720)  # x, y, w, h
        self.installEventFilter(self)
        self.initUI()
        self.show()

    def initUI(self):
        """Creating and setting up the user interface"""
        self.showMaximized()

        # Video open
        self.OpenBTN = QPushButton("Open Video")
        self.OpenBTN.clicked.connect(self.openNewFile)

        # Video properties
        self.FileNameLBL = QLabel()
        self.ResolutionLBL = QLabel()
        self.FpsLBL = QLabel()
        self.LengthLBL = QLabel()

        # Layout for properties
        PropLayout = QVBoxLayout()
        PropLayout.addWidget(self.FileNameLBL)
        PropLayout.addWidget(self.ResolutionLBL)
        PropLayout.addWidget(self.FpsLBL)
        PropLayout.addWidget(self.LengthLBL)

        # Properties groupbox
        self.PropGB = QGroupBox()
        self.PropGB.setTitle("Video properties")
        self.PropGB.setLayout(PropLayout)
        self.PropGB.setVisible(False)

        # Tracking section start and end
        self.setresetStartBTN = QPushButton()
        self.setresetStartBTN.setText("Set")
        self.setresetStartBTN.setMaximumWidth(60)
        self.setresetStartBTN.clicked.connect(self.setresetStart)
        self.setresetStopBTN = QPushButton()
        self.setresetStopBTN.setText("Set")
        self.setresetStopBTN.setMaximumWidth(60)
        self.setresetStopBTN.clicked.connect(self.setresetStop)

        self.SectionStartLBL = QLabel()
        self.SectionStartLBL.setText("Start: - ")
        self.SectionStopLBL = QLabel()
        self.SectionStopLBL.setText("Stop: - ")

        # Tracking section layouts
        SectionStartLayout = QHBoxLayout()
        SectionStartLayout.addWidget(self.SectionStartLBL)
        SectionStartLayout.addWidget(self.setresetStartBTN)

        SectionStopLayout = QHBoxLayout()
        SectionStopLayout.addWidget(self.SectionStopLBL)
        SectionStopLayout.addWidget(self.setresetStopBTN)

        TrackingSectionLayout = QVBoxLayout()
        TrackingSectionLayout.addLayout(SectionStartLayout)
        TrackingSectionLayout.addLayout(SectionStopLayout)
        self.TrackingSectionGB = QGroupBox()
        self.TrackingSectionGB.setTitle("Section to track")
        self.TrackingSectionGB.setLayout(TrackingSectionLayout)
        self.TrackingSectionGB.setFixedWidth(200)

        # Adding points to track
        self.NewObjectBTN = QPushButton()
        self.NewObjectBTN.setText("Add new point")
        self.NewObjectBTN.setFixedHeight(50)
        self.NewObjectBTN.clicked.connect(self.addNewObject)

        self.NameLNE = QLineEdit()
        self.NameLNE.setVisible(False)
        self.PickPointBTN = QPushButton()
        self.PickPointBTN.setText("Point")
        self.PickPointBTN.clicked.connect(self.pickPointStart)
        self.PickPointBTN.setVisible(False)
        self.PickPointBTN.setCheckable(True)

        self.PickRectangleBTN = QPushButton()
        self.PickRectangleBTN.setText("Rectangle")
        self.PickRectangleBTN.clicked.connect(self.pickRectStart)
        self.PickRectangleBTN.setVisible(False)
        self.PickRectangleBTN.setCheckable(True)

        self.SaveBTN = QPushButton()
        self.SaveBTN.setText("Save")
        self.SaveBTN.clicked.connect(self.saveObject)
        self.SaveBTN.setVisible(False)

        self.CancelBTN = QPushButton()
        self.CancelBTN.setText("Cancel")
        self.CancelBTN.clicked.connect(self.cancelObject)
        self.CancelBTN.setVisible(False)

        self.PickLayout = QHBoxLayout()
        self.PickLayout.addWidget(self.PickPointBTN)
        self.PickLayout.addWidget(self.PickRectangleBTN)

        # displaying existing objects
        self.ObjectLWG = ObjectListWidget()
        self.ObjectLWG.delete.connect(lambda name: self.deleteObject(name))
        self.ObjectLWG.changeVisibility.connect(
            lambda name: self.changeObjectDisplay(name)
        )

        self.ObjectsLayout = QVBoxLayout()
        self.ObjectsLayout.addWidget(self.NewObjectBTN)
        self.ObjectsLayout.addWidget(self.NameLNE)
        self.ObjectsLayout.addLayout(self.PickLayout)
        self.ObjectsLayout.addWidget(self.SaveBTN)
        self.ObjectsLayout.addWidget(self.CancelBTN)
        self.ObjectsLayout.addWidget(self.ObjectLWG)

        self.ObjectsGB = QGroupBox()
        self.ObjectsGB.setTitle("Objects to track")
        self.ObjectsGB.setLayout(self.ObjectsLayout)
        self.ObjectsGB.setFixedWidth(200)
        self.ObjectsGB.setFixedHeight(200)

        # Video player controller buttons
        self.StartPauseBTN = QPushButton()
        self.StartPauseBTN.setCheckable(True)
        self.StartPauseBTN.setIcon(QIcon("images/play.svg"))
        self.StartPauseBTN.setMinimumSize(QSize(40, 40))
        self.StartPauseBTN.clicked.connect(self.StartPauseVideo)

        StopBTN = QPushButton()
        StopBTN.setIcon(QIcon("images/stop.svg"))
        StopBTN.setMinimumSize(QSize(40, 40))
        StopBTN.clicked.connect(self.closeVideo)

        self.ForwardBTN = QPushButton()
        self.ForwardBTN.setIcon(QIcon("images/forward.svg"))
        self.ForwardBTN.setMinimumSize(QSize(40, 40))
        self.ForwardBTN.clicked.connect(self.JumpForward)

        self.BackwardBTN = QPushButton()
        self.BackwardBTN.setIcon(QIcon("images/backward.svg"))
        self.BackwardBTN.setMinimumSize(QSize(40, 40))
        self.BackwardBTN.clicked.connect(self.JumpBackward)

        self.VideoSLD = QSlider(Qt.Horizontal)
        self.VideoSLD.setMinimum(0)
        self.VideoSLD.setMaximum(10000)
        self.VideoSLD.setValue(0)
        self.VideoSLD.valueChanged.connect(self.positionVideo)

        # Add controllers to layout
        PlayerControlLayout = QHBoxLayout()
        PlayerControlLayout.addWidget(self.StartPauseBTN)
        PlayerControlLayout.addWidget(StopBTN)
        PlayerControlLayout.addWidget(self.BackwardBTN)
        PlayerControlLayout.addWidget(self.ForwardBTN)
        PlayerControlLayout.addWidget(self.VideoSLD)

        # Zoom and move control buttons
        MoveUpBTN = QPushButton()
        MoveUpBTN.setIcon(QIcon("images/up.svg"))
        MoveUpBTN.setFixedSize(QSize(40, 40))
        MoveUpBTN.setAutoRepeat(True)
        MoveUpBTN.setAutoRepeatDelay(10)
        MoveUpBTN.setAutoRepeatInterval(10)
        MoveUpBTN.clicked.connect(lambda: self.changeYoffset(-1))

        MoveDownBTN = QPushButton()
        MoveDownBTN.setIcon(QIcon("images/down.svg"))
        MoveDownBTN.setFixedSize(QSize(40, 40))
        MoveDownBTN.setAutoRepeat(True)
        MoveDownBTN.setAutoRepeatDelay(10)
        MoveDownBTN.setAutoRepeatInterval(10)
        MoveDownBTN.clicked.connect(lambda: self.changeYoffset(1))

        MoveLeftBTN = QPushButton()
        MoveLeftBTN.setIcon(QIcon("images/left.svg"))
        MoveLeftBTN.setFixedSize(QSize(40, 40))
        MoveLeftBTN.setAutoRepeat(True)
        MoveLeftBTN.setAutoRepeatDelay(10)
        MoveLeftBTN.setAutoRepeatInterval(10)
        MoveLeftBTN.clicked.connect(lambda: self.changeXoffset(-1))

        MoveRightBTN = QPushButton()
        MoveRightBTN.setIcon(QIcon("images/right.svg"))
        MoveRightBTN.setAutoRepeat(True)
        MoveRightBTN.setAutoRepeatDelay(10)
        MoveRightBTN.setAutoRepeatInterval(10)
        MoveRightBTN.setFixedSize(QSize(40, 40))
        MoveRightBTN.clicked.connect(lambda: self.changeXoffset(1))

        # Zoom In/Out
        ZoomInBTN = QPushButton()
        ZoomInBTN.setIcon(QIcon("images/zoom-in.svg"))
        ZoomInBTN.setFixedSize(QSize(40, 70))
        ZoomInBTN.clicked.connect(lambda: self.changeZoom(-0.1))

        ZoomOutBTN = QPushButton()
        ZoomOutBTN.setIcon(QIcon("images/zoom-out.svg"))
        ZoomOutBTN.setFixedSize(QSize(40, 70))
        ZoomOutBTN.clicked.connect(lambda: self.changeZoom(0.1))

        # Label for video frames
        self.VidLBL = VideoLabel()
        self.VidLBL.setAlignment(Qt.AlignCenter)
        self.VidLBL.setMinimumSize(640, 480)
        self.VidLBL.setStyleSheet("background-color: #c9c9c9")
        self.VidLBL.setPixmap(QPixmap("images/video.svg"))

        # Label for timestap
        self.VidTimeLBL = QLabel()
        self.VidTimeLBL.setText("-/-")
        self.VidTimeLBL.setAlignment(Qt.AlignCenter)
        self.VidTimeLBL.setStyleSheet("font-weight: bold; font-size: 14px")
        self.VidTimeLBL.setFixedWidth(200)

        # Add to grid layout
        MoveGridLayout = QGridLayout()
        MoveGridLayout.addWidget(MoveUpBTN, 0, 1)
        MoveGridLayout.addWidget(MoveLeftBTN, 1, 0)
        MoveGridLayout.addWidget(MoveRightBTN, 1, 2)
        MoveGridLayout.addWidget(MoveDownBTN, 2, 1)

        # Add to layout
        ZoomInOutLayout = QVBoxLayout()
        ZoomInOutLayout.addWidget(ZoomInBTN)
        ZoomInOutLayout.addWidget(ZoomOutBTN)

        # Veritcal layout
        ZoomControlLayout = QHBoxLayout()
        ZoomControlLayout.addLayout(MoveGridLayout)
        ZoomControlLayout.addLayout(ZoomInOutLayout)

        self.ZoomControlGB = QGroupBox()
        self.ZoomControlGB.setTitle("Zoom control")
        self.ZoomControlGB.setLayout(ZoomControlLayout)
        self.ZoomControlGB.setFixedWidth(200)

        self.mmLNE = QLineEdit()
        self.mmLNE.setValidator(QDoubleValidator(0, 10000, 2))
        self.mmLNE.setVisible(False)
        self.mmLBL = QLabel("mm")
        self.mmLBL.setVisible(False)

        self.setRulerBTN = QPushButton()
        self.setRulerBTN.setText("Set")
        self.setRulerBTN.clicked.connect(self.setRuler)

        self.saveRulerBTN = QPushButton()
        self.saveRulerBTN.setText("Save")
        self.saveRulerBTN.clicked.connect(self.saveRuler)
        self.saveRulerBTN.setVisible(False)

        self.removeRulerBTN = QPushButton()
        self.removeRulerBTN.setText("Delete")
        self.removeRulerBTN.setVisible(False)
        self.removeRulerBTN.clicked.connect(self.removeRuler)

        self.changeRulerVisibilityBTN = QPushButton()
        self.changeRulerVisibilityBTN.setText("Hide")
        self.changeRulerVisibilityBTN.setVisible(False)
        self.changeRulerVisibilityBTN.clicked.connect(self.changeRulerVisibility)

        # Ruler layout
        self.RulerInputLayout = QHBoxLayout()
        self.RulerInputLayout.addWidget(self.mmLNE)
        self.RulerInputLayout.addWidget(self.mmLBL)
        RulerControlLayout = QHBoxLayout()
        RulerControlLayout.addWidget(self.removeRulerBTN)
        RulerControlLayout.addWidget(self.changeRulerVisibilityBTN)
        RulerLayout = QVBoxLayout()
        RulerLayout.addLayout(self.RulerInputLayout)
        RulerLayout.addWidget(self.setRulerBTN)
        RulerLayout.addWidget(self.saveRulerBTN)
        RulerLayout.addLayout(RulerControlLayout)

        # Ruler groupbox
        self.RulerGB = QGroupBox()
        self.RulerGB.setTitle("Ruler")
        self.RulerGB.setLayout(RulerLayout)
        self.RulerGB.setFixedWidth(200)

        # Select roi
        self.setRoiBTN = QPushButton("Set")
        self.setRoiBTN.clicked.connect(self.setRoi)

        self.saveRoiBTN = QPushButton("Save")
        self.saveRoiBTN.clicked.connect(self.saveRoi)
        self.saveRoiBTN.setVisible(False)

        self.delRoiBTN = QPushButton("Delete")
        self.delRoiBTN.clicked.connect(self.delRoi)
        self.delRoiBTN.setVisible(False)

        roiLayout = QVBoxLayout()
        roiLayout.addWidget(self.setRoiBTN)
        roiLayout.addWidget(self.saveRoiBTN)
        roiLayout.addWidget(self.delRoiBTN)

        self.roiGB = QGroupBox()
        self.roiGB.setTitle("Select ROI rectangle")
        self.roiGB.setLayout(roiLayout)
        self.roiGB.setFixedWidth(200)

        # Track button
        self.TrackBTN = QPushButton()
        self.TrackBTN.setFixedSize(QSize(200, 40))
        self.TrackBTN.setText("Track")
        self.TrackBTN.setEnabled(False)
        self.TrackBTN.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.TrackBTN.clicked.connect(self.showTrackingSettings)

        # Rotation
        addRotBTN = QPushButton("Track rotation")

        # Export and plot button
        self.exportBTN = QPushButton("Plot - Export")
        self.exportBTN.clicked.connect(self.showExportDialog)
        self.exportBTN.setEnabled(False)

        # Video export
        self.exportVidBTN = QPushButton("Export Video")
        self.exportVidBTN.clicked.connect(self.exportVideo)
        self.exportVidBTN.setEnabled(False)

        # Reset all button
        self.resetAllBTN = QPushButton("Reset ALL")
        self.resetAllBTN.clicked.connect(self.resetAll)
        self.exportBTN.setEnabled(False)

        # LSideLayout
        LSideLayout = QVBoxLayout()
        LSideLayout.addWidget(self.OpenBTN)
        LSideLayout.addWidget(self.PropGB)
        LSideLayout.addWidget(self.TrackingSectionGB)
        LSideLayout.addWidget(self.ObjectsGB)
        LSideLayout.addWidget(self.ZoomControlGB)
        LSideLayout.addWidget(self.RulerGB)
        LSideLayout.addWidget(self.roiGB)
        LSideLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        LSideLayout.addWidget(self.TrackBTN)

        RSideLayout = QVBoxLayout()
        RSideLayout.addWidget(self.exportBTN)
        RSideLayout.addWidget(self.exportVidBTN)
        RSideLayout.addWidget(self.resetAllBTN)
        RSideLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        RSideLayout.addWidget(self.VidTimeLBL)
        RSideLayout.setContentsMargins(0, 0, 0, 12)

        # Vertical layout to wrap the player
        VideoControlLayout = QVBoxLayout()
        VideoControlLayout.addWidget(self.VidLBL)
        VideoControlLayout.addLayout(PlayerControlLayout)

        # Horizontal layout with zoom added
        PlayerLayout = QHBoxLayout()
        PlayerLayout.addLayout(LSideLayout)
        PlayerLayout.addLayout(VideoControlLayout)
        PlayerLayout.addLayout(RSideLayout)

        # initializing dialogs
        self.settingsDialog = TrackingSettings()
        self.progressDialog = TrackingProgress()
        self.exportDialog = ExportDialog()
        # self.PlotDialog = PlotDialog()

        # Overall layout
        self.setLayout(PlayerLayout)

    def openNewFile(self):
        """Opens a new video file"""
        filename = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "./",
            "MP4 file (*.mp4);;MOV file (*.mov);;AVI file (*.avi);; MKV file (*.mkv)",
        )[0]
        if filename != "":
            self.filename = filename
            self.openVideo()

    def openVideo(self):
        """Creates the VideoCapture object and gets required properties"""
        self.camera = cv2.VideoCapture(self.filename)

        # important properties
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.num_of_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # display properties
        self.FileNameLBL.setText(f"Filename: {QFileInfo(self.filename).fileName()}")
        self.ResolutionLBL.setText(
            f"Resolution: {self.video_width}x{self.video_height}"
        )
        self.FpsLBL.setText(f"FPS: {self.fps:.2f}")
        self.LengthLBL.setText(
            f"Video length: {self.time_to_display(self.num_of_frames)}"
        )
        self.PropGB.setVisible(True)
        self.OpenBTN.setVisible(False)
        self.VidLBL.wheel.connect(self.changeZoom)

        self.timer.timeout.connect(self.nextFrame)
        self.nextFrame()

    def closeVideo(self):
        """Closes the video, releases the cam"""
        if self.camera is not None:
            self.camera.release()
        self.PropGB.setVisible(False)
        self.VidTimeLBL.setText("-/-")
        self.VidLBL.setPixmap(QPixmap("images/video.svg"))
        try:
            self.VidLBL.disconnect()
        except:
            pass
        self.OpenBTN.setVisible(True)
        self.ObjectLWG.clear()
        self.removeRuler()
        self.cancelObject()
        self.ObjectLWG.clear()
        self.settingsDialog.rotationSettings.p1CMB.clear()
        self.settingsDialog.rotationSettings.p2CMB.clear()
        self.exportDialog.delete_object("ALL")
        self.exportDialog.delete_object("ALL")

        # reset stored properties
        self.camera = None
        self.fps = None
        self.num_of_frames = 0
        self.x_offset = 0
        self.y_offset = 0
        self.zoom = 1  # from 1 goes down to 0
        self.filename = ""
        self.video_width = None
        self.video_height = None
        self.objects_to_track = []
        self.rotations = []
        self.point_tmp = None
        self.rect_tmp = None  # x0, y0, x1, y1
        self.section_stop = None
        self.SectionStopLBL.setText("Stop: - ")
        self.setresetStopBTN.setText("Set")
        self.section_start = None
        self.SectionStartLBL.setText("Stop: - ")
        self.setresetStartBTN.setText("Set")

    def StartPauseVideo(self):
        """Starts and pauses the video"""

        if self.camera is None:
            self.StartPauseBTN.setChecked(False)
            return
        if self.StartPauseBTN.isChecked():
            if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.num_of_frames:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.section_start is not None and self.section_stop is not None:
                if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.section_stop:
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

            self.timer.start(round(1000 / self.fps))
            self.StartPauseBTN.setIcon(QIcon("images/pause.svg"))
        else:
            self.timer.stop()
            self.StartPauseBTN.setIcon(QIcon("images/play.svg"))

    def display_objects(self, frame, pos, section_start, section_stop):
        for obj in self.objects_to_track:
            if obj.visible:
                if (pos >= self.section_start) and (pos <= self.section_stop):
                    if pos == self.section_start:
                        x, y = obj.point
                        frame = cv2.drawMarker(
                            frame, (x, y), (0, 0, 255), 0, thickness=2
                        )
                        x0, y0, x1, y1 = tracker2gui(obj.rectangle)
                        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            obj.name,
                            (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        x, y = obj.point_path[int(pos - self.section_start)]
                        frame = cv2.drawMarker(
                            frame, (int(x), int(y)), (0, 0, 255), 0, thickness=2
                        )
                        x, y = (
                            obj.position[int(pos - self.section_start)][0],
                            obj.position[int(pos - self.section_start)][1],
                        )
                        frame = cv2.drawMarker(
                            frame, (int(x), int(y)), (0, 255, 255), 0, thickness=2,
                        )
                        x0, y0, x1, y1 = tracker2gui(
                            obj.rectangle_path[int(pos - self.section_start - 1)]
                        )
                    # print(f"{x0} {y0} {x1} {y1}")
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        obj.name,
                        (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
        return frame

    def nextFrame(self):
        """Loads and displays the next frame from the video feed"""

        pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)

        # Disables video playing outside of selected section (if selected)
        if self.section_start is not None and self.section_stop is not None:
            if pos < self.section_start - 1:
                pos = self.section_start
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start - 1)
            if self.section_stop < pos:
                pos = self.section_stop
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_stop)
                slider_pos = round(pos / self.num_of_frames * 10000)
                self.modifyTimestampLBL(pos)
                self.VideoSLD.setValue(slider_pos)
                return
            if self.section_stop == pos:
                self.timer.stop()
                self.StartPauseBTN.setChecked(False)
                self.StartPauseBTN.setIcon(QIcon("images/play.svg"))
                return

        # display
        ret, frame = self.camera.read()
        if ret:

            # ruler
            if self.ruler.displayable() and self.ruler.visible:
                frame = cv2.line(
                    frame,
                    (self.ruler.x0, self.ruler.y0),
                    (self.ruler.x1, self.ruler.y1),
                    (0, 0, 255),
                    2,
                )
                frame = cv2.circle(
                    frame, (self.ruler.x0, self.ruler.y0), 5, (0, 0, 255), -1
                )
                frame = cv2.circle(
                    frame, (self.ruler.x1, self.ruler.y1), 5, (0, 0, 255), -1
                )

            # playback
            if self.mode:
                # print(
                #     f"deg:{self.rotations[0].rotation[int(pos-self.section_start-1)]}"
                # )
                frame = display_objects(
                    frame,
                    pos,
                    self.section_start,
                    self.section_stop,
                    self.objects_to_track,
                )
                # frame = self.display_objects(frame, pos)
                # for obj in self.objects_to_track:
                #    if obj.visible:
                #        if (pos >= self.section_start) and (pos <= self.section_stop):
                #            if pos == self.section_start:
                #                x, y = obj.point
                #                frame = cv2.drawMarker(
                #                    frame, (x, y), (0, 0, 255), 0, thickness=2
                #                )
                #                x0, y0, x1, y1 = tracker2gui(obj.rectangle)
                #                frame = cv2.rectangle(
                #                    frame, (x0, y0), (x1, y1), (255, 0, 0), 2
                #                )
                #                cv2.putText(
                #                    frame,
                #                    obj.name,
                #                    (x0, y0 - 5),
                #                    cv2.FONT_HERSHEY_SIMPLEX,
                #                    0.5,
                #                    (255, 0, 0),
                #                    1,
                #                    cv2.LINE_AA,
                #                )
                #            else:
                #                x, y = obj.point_path[int(pos - self.section_start)]
                #                frame = cv2.drawMarker(
                #                    frame, (int(x), int(y)), (0, 0, 255), 0, thickness=2
                #                )
                #                x, y = (
                #                    obj.position[int(pos - self.section_start)][0],
                #                    obj.position[int(pos - self.section_start)][1],
                #                )
                #                frame = cv2.drawMarker(
                #                    frame,
                #                    (int(x), int(y)),
                #                    (0, 255, 255),
                #                    0,
                #                    thickness=2,
                #                )
                #                x0, y0, x1, y1 = tracker2gui(
                #                    obj.rectangle_path[
                #                        int(pos - self.section_start - 1)
                #                    ]
                #                )
                #            # print(f"{x0} {y0} {x1} {y1}")
                #            frame = cv2.rectangle(
                #                frame, (x0, y0), (x1, y1), (255, 0, 0), 2
                #            )
                #            cv2.putText(
                #                frame,
                #                obj.name,
                #                (x0, y0 - 5),
                #                cv2.FONT_HERSHEY_SIMPLEX,
                #                0.5,
                #                (255, 0, 0),
                #                1,
                #                cv2.LINE_AA,
                #            )

            frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * frame.shape[2],
                QImage.Format_RGB888,
            )
            pix = QPixmap.fromImage(img)
            pix = pix.scaled(
                self.VidLBL.width(),
                self.VidLBL.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.VidLBL.setPixmap(pix)

            # move the slider
            pos_in_frames = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            slider_pos = round(pos_in_frames / self.num_of_frames * 10000)
            self.modifyTimestampLBL(pos_in_frames)
            self.VideoSLD.blockSignals(True)
            self.VideoSLD.setValue(slider_pos)
            self.VideoSLD.blockSignals(False)

        else:
            self.timer.stop()
            self.StartPauseBTN.setChecked(False)
            self.StartPauseBTN.setIcon(QIcon("images/play.svg"))

    def ReloadCurrentFrame(self):
        """Reloads and displays the current frame"""

        if self.camera is not None:
            ret, frame = self.camera.retrieve()
            if ret:
                # some kind of gate needed
                if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.section_start:
                    # previous
                    for obj in self.objects_to_track:
                        if obj.visible:
                            x, y = obj.point
                            frame = cv2.drawMarker(
                                frame, (x, y), (0, 0, 255), 0, thickness=2
                            )
                            x0, y0, x1, y1 = tracker2gui(obj.rectangle)
                            frame = cv2.rectangle(
                                frame, (x0, y0), (x1, y1), (255, 0, 0), 2
                            )
                            cv2.putText(
                                frame,
                                obj.name,
                                (x0, y0 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    # current
                    if self.point_tmp is not None:
                        x, y = self.point_tmp
                        frame = cv2.drawMarker(
                            frame, (x, y), (0, 0, 255), 0, thickness=2
                        )
                    if self.rect_tmp is not None:
                        x0, y0, x1, y1 = self.rect_tmp
                        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

                # ruler
                if self.ruler.displayable() and self.ruler.visible:
                    frame = cv2.line(
                        frame,
                        (self.ruler.x0, self.ruler.y0),
                        (self.ruler.x1, self.ruler.y1),
                        (0, 0, 255),
                        2,
                    )
                    frame = cv2.circle(
                        frame, (self.ruler.x0, self.ruler.y0), 5, (0, 0, 255), -1
                    )
                    frame = cv2.circle(
                        frame, (self.ruler.x1, self.ruler.y1), 5, (0, 0, 255), -1
                    )
                if self.roi_rect is not None:
                    x0, y0, x1, y1 = self.roi_rect
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

                frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(
                    frame,
                    frame.shape[1],
                    frame.shape[0],
                    frame.shape[1] * frame.shape[2],
                    QImage.Format_RGB888,
                )
                pix = QPixmap.fromImage(img)
                pix = pix.scaled(
                    self.VidLBL.width(),
                    self.VidLBL.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.VidLBL.setPixmap(pix)

    def JumpForward(self):
        "Jumps formard 10 frames"
        if self.camera is None:
            return
        pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES) + 10
        if pos > self.num_of_frames:
            pos = self.num_of_frames - 1
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.nextFrame()

    def JumpBackward(self):
        "Jumps back 10 frames"
        if self.camera is None:
            return
        pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES) - 10
        if pos < 0:
            pos = 0
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.nextFrame()

    def positionVideo(self):
        """Positions the video according to the Slider value"""
        if self.camera is None:
            return
        pos = self.VideoSLD.value()
        pos_in_frames = round(pos / 10000 * self.num_of_frames)
        """
        if self.section_start is not None:
            if pos_in_frames < self.section_start:
                pos_in_frames = self.section_start  # - 1
                self.VideoSLD.setValue(
                    floor(10000 * pos_in_frames / self.num_of_frames)
                )
        if self.section_stop is not None:
            if pos_in_frames > self.section_stop:
                pos_in_frames = self.section_stop
                self.VideoSLD.setValue(
                    floor(10000 * pos_in_frames / self.num_of_frames)
                )"""
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos_in_frames)
        self.nextFrame()

    def resizeEvent(self, e):
        "Handles the resizing od the window"
        self.ReloadCurrentFrame()
        return super().resizeEvent(e)

    def changeXoffset(self, delta):
        """Adjusts the X offset of the video displaying label"""
        if self.camera is None:
            return
        x_new = self.x_offset + delta
        if ceil(x_new + self.zoom * self.video_width / 2) >= floor(
            self.video_width / 2
        ):
            x_new = floor(self.video_width / 2 - self.zoom * self.video_width / 2)
        elif floor(self.video_width / 2) + x_new <= ceil(
            self.zoom * self.video_width / 2
        ):
            x_new = -floor(self.video_width / 2 - self.zoom * self.video_width / 2)
        self.x_offset = x_new
        self.ReloadCurrentFrame()

    def changeYoffset(self, delta):
        """Adjusts the Y offset of the video displaying label"""
        if self.camera is None:
            return
        y_new = self.y_offset + delta
        if ceil(y_new + self.zoom * self.video_height / 2) >= floor(
            self.video_height / 2
        ):
            y_new = floor(self.video_height / 2 - self.zoom * self.video_height / 2)

        elif floor(self.video_height / 2) + y_new <= ceil(
            self.zoom * self.video_height / 2
        ):
            y_new = -floor(self.video_height / 2 - self.zoom * self.video_height / 2)
        self.y_offset = y_new
        self.ReloadCurrentFrame()

    def changeZoom(self, delta):
        """Zooms in or out of the frame, adjusts the offsets accordingly"""
        if self.camera is None:
            return

        # changing the zoom
        z = self.zoom + delta
        if z <= 0.1:
            z = 0.1
        elif z >= 1:
            z = 1
        self.zoom = z

        # adjusting offset if needed
        if ceil(self.x_offset + self.zoom * self.video_width / 2) >= floor(
            self.video_width / 2
        ):
            self.x_offset = floor(
                self.video_width / 2 - self.zoom * self.video_width / 2
            )
        elif floor(self.video_width / 2) + self.x_offset <= ceil(
            self.zoom * self.video_width / 2
        ):
            self.x_offset = -floor(
                self.video_width / 2 - self.zoom * self.video_width / 2
            )

        if ceil(self.y_offset + self.zoom * self.video_height / 2) >= floor(
            self.video_height / 2
        ):
            self.y_offset = floor(
                self.video_height / 2 - self.zoom * self.video_height / 2
            )

        elif floor(self.video_height / 2) + self.y_offset <= ceil(
            self.zoom * self.video_height / 2
        ):
            self.y_offset = -floor(
                self.video_height / 2 - self.zoom * self.video_height / 2
            )

        self.ReloadCurrentFrame()

    def relative_to_cv(self, x_rel, y_rel):
        """Calculates the relative coordinates emitted by VidLBL to OpenCV coordinates"""
        x = (
            self.x_offset
            + round(x_rel * self.video_width * self.zoom)
            + round(self.video_width / 2)
        )
        y = (
            self.y_offset
            + round(y_rel * self.video_height * self.zoom)
            + round(self.video_height / 2)
        )
        return x, y

    def modifyTimestampLBL(self, index):
        """Adjusts the displayed timestamp/video length"""
        self.VidTimeLBL.setText(
            self.time_to_display(index) + "/" + self.time_to_display(self.num_of_frames)
        )

    def time_to_display(self, index):
        """Calculates and formats the timesptamp from the givan fram index"""
        current_time = index / self.fps
        h = int(current_time / 3600)
        min = int(current_time / 60) - 60 * h
        s = (current_time) - 60 * min
        return (
            str(h).zfill(2) + ":" + str(min).zfill(2) + ":" + str(f"{s:2.2f}").zfill(5)
        )

    def setresetStart(self):
        """Sets and resets the start of the tracked section"""
        if self.camera is None:
            return
        if self.section_start is None:
            self.section_start = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            self.SectionStartLBL.setText(
                f"Start: {self.time_to_display(self.section_start)}"
            )
            self.setresetStartBTN.setText("Clear")
        else:
            self.section_start = None
            self.SectionStartLBL.setText("Start: - ")
            self.setresetStartBTN.setText("Set")

        # check if tracking is enabled
        if (
            len(self.objects_to_track) > 0
            and self.section_start is not None
            and self.section_stop is not None
        ):
            self.TrackBTN.setEnabled(True)
        else:
            self.TrackBTN.setEnabled(False)

    def setresetStop(self):
        """Sets and resets the end of the tracked section"""
        if self.camera is None:
            return
        if self.section_stop is None:
            self.section_stop = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            self.SectionStopLBL.setText(
                f"Stop: {self.time_to_display(self.section_stop)}"
            )
            self.setresetStopBTN.setText("Clear")
        else:
            self.section_stop = None
            self.SectionStopLBL.setText("Stop: - ")
            self.setresetStopBTN.setText("Set")

        # check if tracking is enabled
        if (
            len(self.objects_to_track) > 0
            and self.section_start is not None
            and self.section_stop is not None
        ):
            self.TrackBTN.setEnabled(True)
        else:
            self.TrackBTN.setEnabled(False)

    def addNewObject(self):
        """Cunfigures the GUI for setting the point and tectangle that will be used for the tracking"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return

        # reorganize the layout
        self.NewObjectBTN.setVisible(False)
        self.NameLNE.setVisible(True)
        self.PickPointBTN.setVisible(True)
        self.PickRectangleBTN.setVisible(True)
        self.SaveBTN.setVisible(True)
        self.CancelBTN.setVisible(True)

        self.camera.set(cv2.CAP_PROP_POS_FRAMES, (self.section_start - 1))
        self.nextFrame()
        self.ReloadCurrentFrame()
        self.VideoSLD.setEnabled(False)
        self.StartPauseBTN.setEnabled(False)
        self.ForwardBTN.setEnabled(False)
        self.BackwardBTN.setEnabled(False)

    def saveObject(self):
        """Saves temporary point and rectangle data in the objects_to_track list"""
        name = self.NameLNE.text()
        if self.point_tmp is None or self.rect_tmp is None or name == "":
            return
        # save object
        M = Motion(name, self.point_tmp, gui2tracker(self.rect_tmp))
        self.objects_to_track.append(M)

        # delete temp buffer
        self.point_tmp = None
        self.rect_tmp = None

        # disconnect
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass

        # display in listwidget
        self.ObjectLWG.addItem(M.name)

        self.settingsDialog.rotationSettings.p1CMB.addItem(M.name)
        self.settingsDialog.rotationSettings.p2CMB.addItem(M.name)
        self.exportDialog.add_object(M.name)

        # reorganize widgets
        self.NameLNE.setVisible(False)
        self.NameLNE.clear()
        self.PickPointBTN.setVisible(False)
        self.PickPointBTN.setChecked(False)
        self.PickRectangleBTN.setVisible(False)
        self.PickRectangleBTN.setChecked(False)
        self.SaveBTN.setVisible(False)
        self.CancelBTN.setVisible(False)
        self.NewObjectBTN.setVisible(True)

        # enable video playback
        self.StartPauseBTN.setEnabled(True)
        self.VideoSLD.setEnabled(True)
        self.ForwardBTN.setEnabled(True)
        self.BackwardBTN.setEnabled(True)

        # check if tracking is enabled
        if (
            len(self.objects_to_track) > 0
            and self.section_start is not None
            and self.section_stop is not None
        ):
            self.TrackBTN.setEnabled(True)
        else:
            self.TrackBTN.setEnabled(False)

        # reload
        self.ReloadCurrentFrame()

    def cancelObject(self):
        """Cancels the adding operation, deletes all the buffers and reorganizes the layout"""
        # delete temp buffer
        self.point_tmp = None
        self.rect_tmp = None

        # disconnect
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass

        # reorganize widgets
        self.NameLNE.setVisible(False)
        self.NameLNE.clear()
        self.PickPointBTN.setVisible(False)
        self.PickPointBTN.setChecked(False)
        self.PickRectangleBTN.setVisible(False)
        self.PickRectangleBTN.setChecked(False)
        self.SaveBTN.setVisible(False)
        self.CancelBTN.setVisible(False)
        self.NewObjectBTN.setVisible(True)

        # enable video playback
        self.StartPauseBTN.setEnabled(True)
        self.VideoSLD.setEnabled(True)
        self.ForwardBTN.setEnabled(True)
        self.BackwardBTN.setEnabled(True)
        self.ReloadCurrentFrame()

    def pickPointStart(self):
        """Waits for a signal generated by VidLBL with the coordinates"""
        if (
            self.camera is None
            or self.section_start is None
            and self.section_stop is None
        ):
            return
        # disconnect all signals
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass
        self.PickRectangleBTN.setChecked(False)
        self.VidLBL.press.connect(lambda x, y: self.savePoint(x, y))

    def pickRectStart(self):
        """Waits for a signal generated by VidLBL with coordinates"""
        if (
            self.camera is None
            or self.section_start is None
            and self.section_stop is None
        ):
            return

            # disconnect all signals
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass

        if self.PickPointBTN.isChecked():
            self.PickPointBTN.setChecked(False)

        if self.PickRectangleBTN.isChecked():
            self.VidLBL.moving.connect(lambda x, y: self.resizeRectangle(x, y))
            self.VidLBL.press.connect(lambda x, y: self.initRectangle(x, y))
            self.VidLBL.release.connect(lambda x, y: self.saveRectangle(x, y))
        else:
            try:
                self.VidLBL.moving.disconnect()
            except:
                pass
            try:
                self.VidLBL.press.disconnect()
            except:
                pass
            try:
                self.VidLBL.release.disconnect()
            except:
                pass

    def initRectangle(self, x, y):
        """Initializes the rectangle when the mouse button is pressed"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        x, y = self.relative_to_cv(x, y)
        self.rect_tmp = (x, y, x, y)

    def resizeRectangle(self, x, y):
        """Updates the rectangle when the mouse is moving"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        x1, y1 = self.relative_to_cv(x, y)
        x0, y0, _, _ = self.rect_tmp
        self.rect_tmp = (x0, y0, x1, y1)
        self.ReloadCurrentFrame()

    def saveRectangle(self, x, y):
        """Saves drawn rectangle"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        x1, y1 = self.relative_to_cv(x, y)
        x0, y0, _, _ = self.rect_tmp
        self.rect_tmp = (x0, y0, x1, y1)
        self.PickRectangleBTN.setChecked(False)
        self.pickRectStart()
        self.ReloadCurrentFrame()

    def savePoint(self, x, y):
        """Saves the coordinates recieved from VidLBL"""
        x, y = self.relative_to_cv(x, y)
        self.point_tmp = (x, y)
        self.ReloadCurrentFrame()
        self.VidLBL.press.disconnect()
        self.PickPointBTN.setChecked(False)

    def deleteObject(self, name):
        """Deletes the selected object from memory"""
        index = next(
            (i for i, item in enumerate(self.objects_to_track) if item.name == name), -1
        )
        del self.objects_to_track[index]
        i = self.ObjectLWG.takeItem(index)
        del i

        self.settingsDialog.rotationSettings.p1CMB.removeItem(index)
        self.settingsDialog.rotationSettings.p2CMB.removeItem(index)
        self.exportDialog.delete_object(name)

        # check if tracking is still available
        if (
            len(self.objects_to_track) == 0
            or self.section_start is None
            or self.section_stop is None
        ):
            self.TrackBTN.setEnabled(False)
        self.ReloadCurrentFrame()

    def changeObjectDisplay(self, name):
        """Changes the visibility of the selected object"""
        index = next(
            (i for i, item in enumerate(self.objects_to_track) if item.name == name), -1
        )
        self.objects_to_track[index].visible = not self.objects_to_track[index].visible
        self.ReloadCurrentFrame()

    def setRuler(self):
        """Reorganises the GUI to make ruler control buttons visible connects mouse movement signals"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        self.VidLBL.press.connect(lambda x, y: self.setRulerStart(x, y))
        self.VidLBL.moving.connect(lambda x, y: self.setRulerMove(x, y))
        self.setRulerBTN.setVisible(False)
        self.saveRulerBTN.setVisible(True)
        self.removeRulerBTN.setVisible(True)
        self.mmLBL.setVisible(True)
        self.mmLNE.setVisible(True)

    def setRulerStart(self, x, y):
        """Captures the starting point of the ruler"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        self.ruler.clear()
        x0, y0 = self.relative_to_cv(x, y)
        self.ruler.setP0(x0, y0)
        self.ReloadCurrentFrame()

    def setRulerMove(self, x, y):
        """Refreshes the endpoint of the ruler"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        self.ruler.rdy = False
        x1, y1 = self.relative_to_cv(x, y)
        self.ruler.setP1(x1, y1)
        self.ReloadCurrentFrame()

    def saveRuler(self):
        """Saves the ruler drawn by the user"""
        input = self.mmLNE.text()
        if not self.ruler.displayable() or input == "":
            return
        self.ruler.mm = float(input)
        self.ruler.calculate()
        self.ReloadCurrentFrame()
        self.saveRulerBTN.setVisible(False)
        self.changeRulerVisibilityBTN.setVisible(True)
        self.mmLNE.setVisible(False)
        self.mmLBL.setText(f"{self.ruler.mm_per_pix:.2f} mm/px")
        self.mmLBL.setAlignment(Qt.AlignCenter)
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass

    def removeRuler(self):
        """Removes the ruler"""
        self.ruler.reset()
        self.setRulerBTN.setVisible(True)
        self.saveRulerBTN.setVisible(False)
        self.mmLBL.setVisible(False)
        self.mmLNE.setVisible(False)
        self.mmLBL.setText("mm")
        self.mmLNE.setText("")
        self.removeRulerBTN.setVisible(False)
        self.changeRulerVisibilityBTN.setVisible(False)
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass
        self.ReloadCurrentFrame()

    def changeRulerVisibility(self):
        """Hides/Shows the ruler in the video"""
        if self.ruler.visible:
            self.ruler.visible = False
            self.changeRulerVisibilityBTN.setText("Show")
        else:
            self.ruler.visible = True
            self.changeRulerVisibilityBTN.setText("Hide")
        self.ReloadCurrentFrame()

    def setRoi(self):
        """Reorganises the GUI to make ruler control buttons visible connects mouse movement signals"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        self.VidLBL.press.connect(lambda x, y: self.setRoiStart(x, y))
        self.VidLBL.moving.connect(lambda x, y: self.setRoiMove(x, y))
        self.setRoiBTN.setVisible(False)
        self.saveRoiBTN.setVisible(True)
        self.delRoiBTN.setVisible(True)

    def setRoiStart(self, x, y):
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        x1, y1 = self.relative_to_cv(x, y)
        self.roi_rect = (x1, y1, x1, y1)
        self.ReloadCurrentFrame()

    def setRoiMove(self, x, y):
        """Refreshes the endpoint of the ruler"""
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return
        x1, y1 = self.relative_to_cv(x, y)
        self.roi_rect = (self.roi_rect[0], self.roi_rect[1], x1, y1)

        self.ReloadCurrentFrame()

    def delRoi(self):
        """Removes the ruler"""
        self.roi_rect = None
        self.setRoiBTN.setVisible(True)
        self.saveRoiBTN.setVisible(False)
        self.delRoiBTN.setVisible(False)
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass
        self.ReloadCurrentFrame()

    def saveRoi(self):
        """Saves the roi drawn by the user"""
        self.ReloadCurrentFrame()
        self.saveRoiBTN.setVisible(False)
        try:
            self.VidLBL.press.disconnect()
        except:
            pass
        try:
            self.VidLBL.moving.disconnect()
        except:
            pass
        try:
            self.VidLBL.release.disconnect()
        except:
            pass

    def showTrackingSettings(self):
        """Displays the dialog with the tracking settings"""
        if self.settingsDialog.exec_():
            # TRACKER_TYPE
            tracker_type = self.settingsDialog.tracker_type()

            # ROTATION
            if self.settingsDialog.rotationCHB.isChecked():
                endpoints = self.settingsDialog.rotationSettings.get_endpoints()
            else:
                endpoints = []

            # SIZE-CHANGE
            if self.settingsDialog.sizeCHB.isChecked():
                size = True
            else:
                size = False

            # FPS
            if self.settingsDialog.fps() != "":
                fps = int(self.settingsDialog.fps())
            else:
                fps = self.fps

            # FILTER
            filter = self.settingsDialog.filterCMB.currentText()
            if filter == "None":
                filter_settings = {"filter": "None"}
            elif filter == "Gaussian":
                filter_settings = {
                    "filter": "Gaussian",
                    "window": int(self.settingsDialog.filterSettings.windowLNE.text()),
                    "sigma": int(self.settingsDialog.filterSettings.sigmaLNE.text()),
                }

            # DERIVATIVE
            derivative_settings = None

            # running the tracker
            self.runTracker(
                tracker_type, size, endpoints, fps, filter_settings, derivative_settings
            )

    def eventFilter(self, source, event):
        """Enables users to change X and Y offsets with the W-A-S-D butttons"""
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_W:
                self.changeYoffset(-5)
            elif key == Qt.Key_A:
                self.changeXoffset(-5)
            elif key == Qt.Key_S:
                self.changeYoffset(5)
            elif key == Qt.Key_D:
                self.changeXoffset(5)
        return super(VideoWidget, self).eventFilter(source, event)

    def pix2mm(self, data):
        if self.ruler.mm_per_pix is not None:
            return data * self.ruler.mm_per_pix

    def pix2m(self, data):
        if self.ruler.mm_per_pix is not None:
            return data * self.ruler.mm_per_pix / 1000

    def runTracker(
        self, tracker_type, size, endpoints, fps, filter_settings, derivative_settings
    ):
        """Runs the seleted tracking algorithm with the help of a QThread"""
        self.tracker = TrackingThread(
            self.objects_to_track,
            self.camera,
            self.section_start,
            self.section_stop,
            tracker_type,
            size,
            endpoints,
            fps,
            filter_settings,
            derivative_settings,
            self.timestamp,
            self.roi_rect,
        )
        self.tracker.progressChanged.connect(self.progressDialog.updateBar)
        self.tracker.newObject.connect(self.progressDialog.updateName)
        self.tracker.success.connect(self.trackingSucceeded)
        self.tracker.error_occured.connect(self.showErrorMessage)
        self.tracker.rotation_calculated.connect(self.saveRotation)
        self.tracker.start()
        self.progressDialog.show()
        self.progressDialog.rejected.connect(self.tracker.cancel)
        self.tracker.finished.connect(self.progressDialog.accept)

    def showErrorMessage(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error occured!")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    def trackingSucceeded(self):
        self.mode = True
        self.OpenBTN.setEnabled(False)
        self.PropGB.setEnabled(False)
        self.TrackingSectionGB.setEnabled(False)
        self.ObjectsGB.setEnabled(False)
        self.ZoomControlGB.setEnabled(False)
        self.RulerGB.setEnabled(False)
        self.exportBTN.setEnabled(True)
        self.exportVidBTN.setEnabled(True)
        self.resetAllBTN.setEnabled(True)

    def resetAll(self):
        # layout reorganization
        self.mode = False
        self.OpenBTN.setEnabled(True)
        self.PropGB.setEnabled(True)
        self.TrackingSectionGB.setEnabled(True)
        self.ObjectsGB.setEnabled(True)
        self.ZoomControlGB.setEnabled(True)
        self.RulerGB.setEnabled(True)
        self.exportBTN.setEnabled(False)
        self.exportVidBTN.setEnabled(False)

        # reset properties
        self.section_start = 0  # to make the reset successful
        self.section_stop = 0
        self.setresetStart()
        self.setresetStop()

        self.objects_to_track = []
        self.rotations = []
        self.timestamp = []

        self.point_tmp = None
        self.rect_tmp = None  # x0, y0, x1, y1

        self.removeRuler()
        self.cancelObject()

        self.roi_rect = None

        self.ObjectLWG.clear()
        self.settingsDialog.rotationSettings.p1CMB.clear()
        self.settingsDialog.rotationSettings.p2CMB.clear()
        self.exportDialog.delete_object("ALL")
        self.exportDialog.delete_object("ALL")

        self.ReloadCurrentFrame()

    def saveRotation(self, rotation):
        self.rotations.append(rotation)
        # reorganize the wideo widget

    def showExportDialog(self):
        self.exportDialog.setRuler(self.ruler.rdy)
        if self.exportDialog.exec_():
            parameters = self.exportDialog.parameters
            # print(parameters)
            if len(parameters) != 0:
                return self.getPlotData(parameters)

    def getPlotData(self, parameters):
        # create the numpy aray for the data
        if not self.mode:
            return

        exp_ok = False
        data = None
        if parameters["mode"] == "MOV" and len(parameters["objects"]):
            data = np.zeros((len(self.timestamp), len(parameters["objects"]) + 1,))
            cols = []

            # get timestamp
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")
            for i in range(len(parameters["objects"])):

                # timestamp only needed once
                obj = self.objects_to_track[i]

                # position
                if parameters["prop"] == "POS":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.position[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = obj.position[:, 1]
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Position"
                elif parameters["prop"] == "VEL":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.position[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = obj.position[:, 1]  # CHANGE IT WHEN READY
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Velocity"

                elif parameters["prop"] == "ACC":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.position[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = obj.position[:, 1]  # CHANGE IT WHEN READY
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Acceletration"

            if parameters["unit"] == "mm":
                data = pix2mm(data, self.ruler.mm_per_pix)
            elif parameters["unit"] == "m":
                data = pix2m(data, self.ruler.mm_per_pix)

        elif parameters["mode"] == "ROT":
            data = np.zeros(
                (
                    len(self.timestamp),
                    len(self.rotations)
                    + 1,  # ezt majd át kell írni len(parameters["rotation"]) ra
                )
            )
            cols = []
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")

            for i in range(len(self.rotations)):
                rot = self.rotations[i]

                if parameters["prop"] == "POS":
                    data[:, i + 1] = rot.rotation
                    cols.append(rot.P1.name + " + " + rot.P2.name + " rotation")
                    title = "Rotation"
                    exp_ok = True
                elif parameters["prop"] == "VEL":
                    data[:, i + 1] = rot.rotation
                    cols.append(rot.P1.name + " + " + rot.P2.name + " angular velocity")
                    title = "Angular velocity"
                    exp_ok = True

                elif parameters["prop"] == "ACC":
                    data[:, i + 1] = rot.rotation
                    cols.append(
                        rot.P1.name + " + " + rot.P2.name + " angular acceleration"
                    )
                    title = "Angular acceleration"
                    exp_ok = True

            if parameters["unit"] == "deg":
                data = rad2deg_(data, self.ruler.mm_per_pix)

        elif parameters["mode"] == "SIZ":
            data = np.zeros((len(self.timestamp), len(parameters["objects"]) + 1,))
            cols = []
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")
            for i in range(len(parameters["objects"])):

                # timestamp only needed once
                obj = self.objects_to_track[i]

                # position
                data[:, i + 1] = np.asarray(obj.size_change)
                cols.append(obj.name + " size change")
                title = "Size change"
                exp_ok = True

        if data is None or not exp_ok:
            return
        df = pd.DataFrame(data, columns=cols)
        if parameters["out"] == "PLOT":
            self.plotter = PlotDialog(df, title, get_unit(parameters))
            self.plotter.show()
        elif parameters["out"] == "EXP":
            save_name = QFileDialog.getSaveFileName(
                self,
                "Save data",
                "/",
                "Text file (*.txt);;CSV file (*.csv);;Excel file (*.xlsx)",
            )
            if save_name[0] != "":
                if save_name[1] == "Excel file (*.xlsx)":
                    df.to_excel(save_name[0])
                else:
                    df.to_csv(save_name[0])

    def exportVideo(self):
        save_name = QFileDialog.getSaveFileName(
            self, "Export Video", "/", "MP4 file (*.mp4)",
        )
        if save_name[0] != "":
            self.exporter = ExportingThread(
                self.camera,
                self.objects_to_track,
                self.section_start,
                self.section_stop,
                save_name[0],
                self.fps,
            )
            self.progressDialog.updateName("Expoting video to " + save_name[0])
            self.progressDialog.rejected.connect(self.exporter.cancel)
            self.exporter.progressChanged.connect(self.progressDialog.updateBar)
            self.exporter.finished.connect(self.progressDialog.accept)
            self.exporter.start()
            self.progressDialog.show()


# run the application
if __name__ == "__main__":
    App = QApplication(sys.argv)
    root = VideoWidget()
    sys.exit(App.exec())

# TODO: replace obj.position(line 1423), changing modes disabling layouts enabling export settings, multiple rotation tracking
