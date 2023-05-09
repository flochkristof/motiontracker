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

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (
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
    QCheckBox,
    QComboBox,
)

from MotionTrackerBeta.widgets.dialogs import *
from MotionTrackerBeta.widgets.lists import *
from MotionTrackerBeta.widgets.video import VideoLabel
from MotionTrackerBeta.widgets.trackers import TrackingThreadV2
from MotionTrackerBeta.widgets.process import PostProcesserThread
from MotionTrackerBeta.widgets.export import ExportingThread

from MotionTrackerBeta.functions.display import display_objects, draw_grid
from MotionTrackerBeta.functions.transforms import *
from MotionTrackerBeta.functions.helper import *

from MotionTrackerBeta.classes.classes import *

from math import floor, ceil

import os

import pandas as pd

import cv2


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()

        self.camera = None  # cv2.VideoCapture object
        self.fps = None  # fps read from file
        self.num_of_frames = 0  # nember of frames in video
        self.x_offset = 0  # x-offset of the zoomed in window
        self.y_offset = 0  # y-offset of the zommed in window
        self.zoom = 1  # rate of zoom: from 1 goes down to 0
        self.section_start = None  # index of the starting frame of the tracked section
        self.section_stop = None  # index of the ending frame of the tracked section
        self.roi_rect = None  # Range-of interest rectangle, if selected
        self.filename = ""  # name of the opened video file
        self.video_width = None  # width of the opened video
        self.video_height = None  # height of the opened video
        self.objects_to_track = []  # list to store the tracked objects
        self.rotations = []  # list to store the tracked rotations
        self.timestamp = []  # list to store the timestamps after the tracking
        self.point_tmp = (
            None  # temporary variable for point selection of the tracked objects
        )
        self.rect_tmp = None  # temporary variable for region selection of the tracked objects (x0, y0, x1, y1)
        self.ruler = Ruler()  # ruler object to store calibration data
        self.mode = False  # False: before tracking, True: after successfull tracking
        self.grid_displayable = False  # To properly display grid

        self.timer = (
            QTimer()
        )  # To play the video, calls the display function of the next frame every tick
        self.timer.setTimerType(Qt.PreciseTimer)

        self.setGeometry(100, 100, 1280, 720)  # Default geometry (x, y, w, h)
        self.installEventFilter(self)  # To register mouse clicks and key presses

        self.initUI()  # initialize UI
        self.showMaximized()  # full window

        self.show()

    def initUI(self):
        """Creates and configures the user interface"""

        # Styling
        self.setObjectName("main_window")
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/main.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        self.setWindowTitle("Motion Tracker Beta")

        # Video open button
        self.OpenBTN = QPushButton("Open Video")
        self.OpenBTN.clicked.connect(self.openNewFile)
        self.OpenBTN.setObjectName("openBTN")

        # Video properties
        self.FileNameLBL = QLabel()
        self.FileNameLBL.setWordWrap(True)
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
        self.PropGB.setFixedWidth(200)

        # Tracking section start and end
        self.setresetStartBTN = QPushButton()
        self.setresetStartBTN.setText("Set")
        self.setresetStartBTN.setMaximumWidth(60)
        self.setresetStartBTN.clicked.connect(self.setresetStart)
        self.setresetStartBTN.setObjectName("setresetStartBTN")
        self.setresetStopBTN = QPushButton()
        self.setresetStopBTN.setText("Set")
        self.setresetStopBTN.setMaximumWidth(60)
        self.setresetStopBTN.clicked.connect(self.setresetStop)
        self.setresetStopBTN.setObjectName("setresetStopBTN")

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

        # Object name
        self.NameLNE = QLineEdit()
        self.NameLNE.setVisible(False)

        # Object point
        self.PickPointBTN = QPushButton()
        self.PickPointBTN.setText("Point")
        self.PickPointBTN.clicked.connect(self.pickPointStart)
        self.PickPointBTN.setVisible(False)
        self.PickPointBTN.setCheckable(True)
        self.PickPointBTN.setObjectName("pointBTN")

        # Object region rectangle
        self.PickRectangleBTN = QPushButton()
        self.PickRectangleBTN.setText("Region")
        self.PickRectangleBTN.clicked.connect(self.pickRectStart)
        self.PickRectangleBTN.setVisible(False)
        self.PickRectangleBTN.setCheckable(True)
        self.PickRectangleBTN.setObjectName("rectBTN")

        # Save object
        self.SaveBTN = QPushButton()
        self.SaveBTN.setText("Save")
        self.SaveBTN.clicked.connect(self.saveObject)
        self.SaveBTN.setVisible(False)
        self.SaveBTN.setObjectName("saveBTN")

        # Canceling object creation
        self.CancelBTN = QPushButton()
        self.CancelBTN.setText("Cancel")
        self.CancelBTN.clicked.connect(self.cancelObject)
        self.CancelBTN.setVisible(False)
        self.CancelBTN.setObjectName("cancelBTN")

        # Order into layout
        self.PickLayout = QHBoxLayout()
        self.PickLayout.addWidget(self.PickPointBTN)
        self.PickLayout.addWidget(self.PickRectangleBTN)

        # Displaying existing objects
        self.ObjectLWG = ObjectListWidget()
        self.ObjectLWG.setObjectName("obj_list")
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
        self.StartPauseBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/play.svg"))
        self.StartPauseBTN.setMinimumSize(QSize(40, 40))
        self.StartPauseBTN.clicked.connect(self.StartPauseVideo)

        # Stops the video, releases the camera, resets everything
        StopBTN = QPushButton()
        StopBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/stop.svg"))
        StopBTN.setMinimumSize(QSize(40, 40))
        StopBTN.clicked.connect(self.closeVideo)

        # Jumping forward in the video
        self.ForwardBTN = QPushButton()
        self.ForwardBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/forward.svg"))
        self.ForwardBTN.setMinimumSize(QSize(40, 40))
        self.ForwardBTN.clicked.connect(self.JumpForward)

        # jumping backward in the video
        self.BackwardBTN = QPushButton()
        self.BackwardBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/backward.svg"))
        self.BackwardBTN.setMinimumSize(QSize(40, 40))
        self.BackwardBTN.clicked.connect(self.JumpBackward)

        # jump to the first frqme of the video
        self.StartBTN = QPushButton()
        self.StartBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/start.svg"))
        self.StartBTN.setMinimumSize(QSize(40, 40))
        self.StartBTN.clicked.connect(self.JumpStart)

        # jump to the last frame of the video
        self.EndBTN = QPushButton()
        self.EndBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/end.svg"))
        self.EndBTN.setMinimumSize(QSize(40, 40))
        self.EndBTN.clicked.connect(self.JumpEnd)

        # slider for easier navigation
        self.VideoSLD = QSlider(Qt.Horizontal)
        self.VideoSLD.setMinimum(0)
        self.VideoSLD.setMaximum(10000)
        self.VideoSLD.setValue(0)
        self.VideoSLD.valueChanged.connect(self.positionVideo)

        # Add controllers to layout
        PlayerControlLayout = QHBoxLayout()
        PlayerControlLayout.addWidget(self.StartBTN)
        PlayerControlLayout.addWidget(self.BackwardBTN)
        PlayerControlLayout.addWidget(self.StartPauseBTN)
        PlayerControlLayout.addWidget(StopBTN)
        PlayerControlLayout.addWidget(self.ForwardBTN)
        PlayerControlLayout.addWidget(self.EndBTN)
        PlayerControlLayout.addWidget(self.VideoSLD)

        # Zoom and move control buttons
        MoveUpBTN = QPushButton()
        MoveUpBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/up.svg"))
        MoveUpBTN.setFixedSize(QSize(40, 40))
        MoveUpBTN.setAutoRepeat(True)
        MoveUpBTN.setAutoRepeatDelay(10)
        MoveUpBTN.setAutoRepeatInterval(10)
        MoveUpBTN.clicked.connect(lambda: self.changeYoffset(-1))

        MoveDownBTN = QPushButton()
        MoveDownBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/down.svg"))
        MoveDownBTN.setFixedSize(QSize(40, 40))
        MoveDownBTN.setAutoRepeat(True)
        MoveDownBTN.setAutoRepeatDelay(10)
        MoveDownBTN.setAutoRepeatInterval(10)
        MoveDownBTN.clicked.connect(lambda: self.changeYoffset(1))

        MoveLeftBTN = QPushButton()
        MoveLeftBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/left.svg"))
        MoveLeftBTN.setFixedSize(QSize(40, 40))
        MoveLeftBTN.setAutoRepeat(True)
        MoveLeftBTN.setAutoRepeatDelay(10)
        MoveLeftBTN.setAutoRepeatInterval(10)
        MoveLeftBTN.clicked.connect(lambda: self.changeXoffset(-1))

        MoveRightBTN = QPushButton()
        MoveRightBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/right.svg"))
        MoveRightBTN.setAutoRepeat(True)
        MoveRightBTN.setAutoRepeatDelay(10)
        MoveRightBTN.setAutoRepeatInterval(10)
        MoveRightBTN.setFixedSize(QSize(40, 40))
        MoveRightBTN.clicked.connect(lambda: self.changeXoffset(1))

        # Zoom In/Out
        ZoomInBTN = QPushButton()
        ZoomInBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/zoom-in.svg"))
        ZoomInBTN.setFixedSize(QSize(40, 70))
        ZoomInBTN.clicked.connect(lambda: self.changeZoom(-0.1))

        ZoomOutBTN = QPushButton()
        ZoomOutBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/zoom-out.svg"))
        ZoomOutBTN.setFixedSize(QSize(40, 70))
        ZoomOutBTN.clicked.connect(lambda: self.changeZoom(0.1))

        # Grid
        GridNumLBL = QLabel("Number of lines:")
        GridNumLBL.setObjectName("GridNumLBL")
        XGridLBL = QLabel("X:")
        YGridLBL = QLabel("Y:")

        self.XGridLNE = QLineEdit()
        self.XGridLNE.setValidator(QIntValidator(1, 500))
        self.XGridLNE.setText("10")
        self.XGridLNE.textChanged.connect(self.GridUpdate)
        self.YGridLNE = QLineEdit()
        self.YGridLNE.setValidator(QIntValidator(1, 500))
        self.YGridLNE.setText("10")
        self.YGridLNE.textChanged.connect(self.GridUpdate)

        self.GridColorCMB = QComboBox()
        self.GridColorCMB.addItems(["black", "white", "red", "green", "blue"])
        self.GridColorCMB.currentTextChanged.connect(self.GridUpdate)

        XLayout = QHBoxLayout()
        XLayout.addWidget(XGridLBL)
        XLayout.addWidget(self.XGridLNE)
        YLayout = QHBoxLayout()
        YLayout.addWidget(YGridLBL)
        YLayout.addWidget(self.YGridLNE)

        GridLayout = QVBoxLayout()
        GridLayout.addWidget(GridNumLBL)
        GridLayout.addLayout(XLayout)
        GridLayout.addLayout(YLayout)
        GridLayout.addWidget(self.GridColorCMB)

        self.GridGB = QGroupBox("Grid")
        self.GridGB.setObjectName("GridGB")
        self.GridGB.setFixedWidth(200)
        self.GridGB.setCheckable(True)
        self.GridGB.setLayout(GridLayout)
        self.GridGB.toggled.connect(self.GridUpdate)

        # Label for video frames
        self.VidLBL = VideoLabel()
        self.VidLBL.setAlignment(Qt.AlignCenter)
        self.VidLBL.setMinimumSize(640, 480)
        self.VidLBL.setStyleSheet("background-color: #c9c9c9")
        self.VidLBL.setPixmap(QPixmap(os.path.dirname(os.path.dirname(__file__))+"/images/video.svg"))

        # Label for timestamp
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

        # Lineedit for ruler input
        self.mmLNE = QLineEdit()
        self.mmLNE.setValidator(QDoubleValidator(0, 10000, 2))
        self.mmLNE.setVisible(False)
        self.mmLBL = QLabel("mm")
        self.mmLBL.setVisible(False)

        # setting the ruler
        self.setRulerBTN = QPushButton()
        self.setRulerBTN.setText("Set")
        self.setRulerBTN.clicked.connect(self.setRuler)
        self.setRulerBTN.setObjectName("setRulerBTN")

        # saving the ruler
        self.saveRulerBTN = QPushButton()
        self.saveRulerBTN.setText("Save")
        self.saveRulerBTN.clicked.connect(self.saveRuler)
        self.saveRulerBTN.setVisible(False)
        self.saveRulerBTN.setObjectName("saveRulerBTN")

        # removing the ruler
        self.removeRulerBTN = QPushButton()
        self.removeRulerBTN.setText("Delete")
        self.removeRulerBTN.setVisible(False)
        self.removeRulerBTN.clicked.connect(self.removeRuler)
        self.removeRulerBTN.setObjectName("removeRulerBTN")

        # hiding displaying the ruler
        self.changeRulerVisibilityBTN = QPushButton()
        self.changeRulerVisibilityBTN.setText("Hide")
        self.changeRulerVisibilityBTN.setVisible(False)
        self.changeRulerVisibilityBTN.clicked.connect(self.changeRulerVisibility)
        self.changeRulerVisibilityBTN.setObjectName("changeRulerVisibilityBTN")

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
        self.setRoiBTN.setObjectName("setRoiBTN")

        # Save roi
        self.saveRoiBTN = QPushButton("Save")
        self.saveRoiBTN.clicked.connect(self.saveRoi)
        self.saveRoiBTN.setVisible(False)
        self.saveRoiBTN.setObjectName("saveRoiBTN")

        # Delete roi
        self.delRoiBTN = QPushButton("Delete")
        self.delRoiBTN.clicked.connect(self.delRoi)
        self.delRoiBTN.setVisible(False)
        self.delRoiBTN.setObjectName("delRoiBTN")

        # ROI into layout & groupbox
        roiLayout = QVBoxLayout()
        roiLayout.addWidget(self.setRoiBTN)
        roiLayout.addWidget(self.saveRoiBTN)
        roiLayout.addWidget(self.delRoiBTN)

        self.roiGB = QGroupBox()
        self.roiGB.setTitle("Select ROI region")
        self.roiGB.setLayout(roiLayout)
        self.roiGB.setFixedWidth(200)

        # recrocess results
        self.reProcessBTN = QPushButton()
        # self.TrackBTN.setFixedSize(QSize(200, 40))
        self.reProcessBTN.setText("Reprocess")
        self.reProcessBTN.setVisible(False)
        self.reProcessBTN.setObjectName("reProcessBTN")
        self.reProcessBTN.clicked.connect(self.openPostProcessing)

        # Track button
        self.TrackBTN = QPushButton()
        # self.TrackBTN.setFixedSize(QSize(200, 40))
        self.TrackBTN.setText("Track")
        self.TrackBTN.setEnabled(False)
        self.TrackBTN.setObjectName("trackBTN")
        self.TrackBTN.clicked.connect(self.showTrackingSettings)

        # Rotation
        addRotBTN = QPushButton("Track rotation")
        addRotBTN.clicked.connect(self.addRotation)
        addRotBTN.setObjectName("addRotBTN")

        # to display existing rotation objects
        self.rotLWG = RotationListWidget()
        self.rotLWG.setObjectName("rot_list")
        self.rotLWG.delete.connect(self.deleteRotation)

        # rotation layout & groupbox
        rotLayout = QVBoxLayout()
        rotLayout.addWidget(addRotBTN)
        rotLayout.addWidget(self.rotLWG)

        self.rotGB = QGroupBox()
        self.rotGB.setTitle("Rotation")
        self.rotGB.setLayout(rotLayout)
        self.rotGB.setEnabled(False)
        self.rotGB.setFixedWidth(200)
        self.rotGB.setFixedHeight(120)

        # Export and plot button
        self.exportBTN = QPushButton("Plot - Export")
        self.exportBTN.clicked.connect(self.showExportDialog)
        self.exportBTN.setEnabled(False)
        self.exportBTN.setObjectName("exportBTN")

        # Video export
        self.exportVidBTN = QPushButton("Export Video")
        self.exportVidBTN.clicked.connect(self.exportVideo)
        self.exportVidBTN.setObjectName("exportVidBTN")
        self.exportVidBTN.setEnabled(False)

        # Playback options
        propertiesLBL = QLabel("Displayed properties:")
        propertiesLBL.setObjectName("propertiesLBL")
        self.boxCHB = QCheckBox("Region")
        self.boxCHB.setChecked(True)
        self.pointCHB = QCheckBox("Point")
        self.pointCHB.setChecked(True)
        # self.trajectoryCHB = QCheckBox("Trajectory")
        trajectoryLBL = QLabel("Trajectory length:")
        trajectoryLBL.setObjectName("trajectoryLBL")
        self.playbackSLD = QSlider(Qt.Horizontal)
        self.playbackSLD.setMinimum(0)
        self.playbackSLD.setMaximum(100)
        self.playbackSLD.setValue(0)

        playbackLayout = QVBoxLayout()
        playbackLayout.addWidget(propertiesLBL)
        playbackLayout.addWidget(self.boxCHB)
        playbackLayout.addWidget(self.pointCHB)
        # playbackLayout.addWidget(self.trajectoryCHB)
        playbackLayout.addWidget(trajectoryLBL)
        playbackLayout.addWidget(self.playbackSLD)
        playbackLayout.addWidget(self.exportVidBTN)

        self.playbackGB = QGroupBox("Playback")
        self.playbackGB.setLayout(playbackLayout)
        self.playbackGB.setFixedWidth(200)
        self.playbackGB.setEnabled(False)

        # Reset all button
        self.resetAllBTN = QPushButton("Reset ALL")

        self.resetAllBTN.clicked.connect(self.resetAll)
        self.resetAllBTN.setObjectName("resetAllBTN")

        # LSideLayout
        LSideLayout = QVBoxLayout()
        LSideLayout.addWidget(self.OpenBTN)
        LSideLayout.addWidget(self.PropGB)
        LSideLayout.addWidget(self.TrackingSectionGB)
        LSideLayout.addWidget(self.ObjectsGB)
        # LSideLayout.addWidget(self.GridGB)
        # LSideLayout.addWidget(self.ZoomControlGB)
        LSideLayout.addWidget(self.RulerGB)
        LSideLayout.addWidget(self.roiGB)
        LSideLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Maximum, QSizePolicy.Expanding)
        )
        LSideLayout.addWidget(self.reProcessBTN)
        LSideLayout.addWidget(self.TrackBTN)

        # RSide layout-> Everything after tracking
        RSideLayout = QVBoxLayout()
        RSideLayout.addWidget(self.rotGB)
        RSideLayout.addWidget(self.exportBTN)
        RSideLayout.addWidget(self.playbackGB)
        # RSideLayout.addWidget(self.exportVidBTN)
        RSideLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Maximum, QSizePolicy.Expanding)
        )
        RSideLayout.addWidget(self.GridGB)
        RSideLayout.addWidget(self.ZoomControlGB)
        RSideLayout.addWidget(self.resetAllBTN)
        RSideLayout.addItem(QSpacerItem(0, 10, QSizePolicy.Maximum))
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
        self.postProcessProgressDialog = CalculationProgress()
        self.postProcessDialog = PostProcessSettings()
        self.exportDialog = ExportDialog()
        self.exportDialog.export.connect(self.getPlotData, Qt.UniqueConnection)
        # self.PlotDialog = PlotDialog()

        # Overall layout
        self.setLayout(PlayerLayout)

    def openNewFile(self):
        """Opens a new video file"""
        filename = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "user/Documents/",
            "MP4 file (*.mp4);;MOV file (*.mov);;AVI file (*.avi);; MKV file (*.mkv)",
        )[0]
        if filename != "":
            self.filename = filename
            self.openVideo()

    def openVideo(self):
        """Creates the VideoCapture object and gets required properties"""
        self.camera = cv2.VideoCapture(self.filename)

        # get essential video properties
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.num_of_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # display the collected data in the gui
        self.FileNameLBL.setText(f"Filename: {QFileInfo(self.filename).fileName()}")
        self.ResolutionLBL.setText(
            f"Resolution: {self.video_width}x{self.video_height}"
        )
        self.FpsLBL.setText(f"FPS: {self.fps:.2f}")
        self.LengthLBL.setText(
            f"Video length: {self.time_to_display(self.num_of_frames)}"
        )

        # reset any previous existing data
        self.resetAll()

        # Reorganize layout
        self.PropGB.setVisible(True)
        self.OpenBTN.setVisible(False)

        # Connect to timer and zoom
        self.VidLBL.wheel.connect(self.changeZoom)
        self.timer.timeout.connect(self.nextFrame)

        # load first frame
        self.nextFrame()

    def closeVideo(self):
        """Closes the video, releases the cam"""

        # disablé timer
        if self.timer.isActive():
            self.StartPauseVideo()
            self.timer.stop()

        # release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # reorganize layout
        self.PropGB.setVisible(False)
        self.VidTimeLBL.setText("-/-")
        self.VidLBL.setPixmap(QPixmap(os.path.dirname(__file__)+"/images/video.svg"))
        # disconnect any signal that may be connected with the VidLBL
        try:
            self.VidLBL.disconnect()
        except:  # required because it raises an exception if no signal was connected
            pass

        # resetting data
        self.OpenBTN.setVisible(True)
        self.resetAll()

    def StartPauseVideo(self):
        """Starts and pauses the video"""

        # check if there is a video opened
        if self.camera is None:
            self.StartPauseBTN.setChecked(False)
            # no popup needed
            return

        # start playing
        if self.StartPauseBTN.isChecked():
            # go back to the beginning if ended
            if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.num_of_frames:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # if section is selected adjust the boundaries
            if self.section_start is not None and self.section_stop is not None:
                if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.section_stop:
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

            # timer that calls the nextFrame method every tick
            self.timer.start(round(1000 / self.fps))

            # chnge icon
            self.StartPauseBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/pause.svg"))
        else:

            # stop the timer and change the icon
            self.timer.stop()
            self.StartPauseBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/play.svg"))

    def nextFrame(self):
        """Loads and displays the next frame from the video feed"""

        # get current position
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
                self.StartPauseBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/play.svg"))
                return

        # Read and display frame
        ret, frame = self.camera.read()
        if ret:
            # Grid
            if self.GridGB.isChecked():
                frame = draw_grid(
                    int(self.XGridLNE.text()),
                    int(self.YGridLNE.text()),
                    frame,
                    self.GridColorCMB.currentText(),
                )

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
                frame = display_objects(
                    frame,
                    pos,
                    self.section_start,
                    self.section_stop,
                    self.objects_to_track,
                    self.boxCHB.isChecked(),
                    self.pointCHB.isChecked(),
                    round(self.playbackSLD.value() * self.num_of_frames / 100),
                )

            # crop and color
            frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # convert form opencv to pyqt
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

            # display label
            self.VidLBL.setPixmap(pix)

            # move the slider
            pos_in_frames = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            slider_pos = round((pos_in_frames) / self.num_of_frames * 10000)
            self.modifyTimestampLBL(pos_in_frames)
            self.VideoSLD.blockSignals(True)
            self.VideoSLD.setValue(slider_pos)
            self.VideoSLD.blockSignals(False)

        else:
            # stop playing
            self.timer.stop()
            self.StartPauseBTN.setChecked(False)
            self.StartPauseBTN.setIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/play.svg"))

    def ReloadCurrentFrame(self):
        """Reloads and displays the current frame"""

        # check if there is a video opened
        if self.camera is None:
            return

        # retrieve current frame
        ret, frame = self.camera.retrieve()
        if ret:
            # Grid
            if self.GridGB.isChecked():
                frame = draw_grid(
                    int(self.XGridLNE.text()),
                    int(self.YGridLNE.text()),
                    frame,
                    self.GridColorCMB.currentText(),
                )

            # First frame of the section
            if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.section_start:
                # display objects
                for obj in self.objects_to_track:
                    if obj.visible:
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
                # Display object selection point
                if self.point_tmp is not None:
                    x, y = self.point_tmp
                    frame = cv2.drawMarker(frame, (x, y), (0, 0, 255), 0, thickness=2)

                # Display object selection rectangle
                if self.rect_tmp is not None:
                    x0, y0, x1, y1 = self.rect_tmp
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # ROI Rectangle
            if self.roi_rect is not None:
                x0, y0, x1, y1 = self.roi_rect
                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

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

            # playback after tracking
            if self.mode:
                frame = display_objects(
                    frame,
                    self.camera.get(cv2.CAP_PROP_POS_FRAMES) - 1,
                    self.section_start,
                    self.section_stop,
                    self.objects_to_track,
                    self.boxCHB.isChecked(),
                    self.pointCHB.isChecked(),
                    round(self.playbackSLD.value() * self.num_of_frames / 100),
                )

            # crop and color
            frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # convert from opencv to pyqt
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

            # display in label
            self.VidLBL.setPixmap(pix)

    def JumpForward(self):
        "Jumps formard 10 frames"

        # check if there is a video opened
        if self.camera is None:
            return

        # get current position
        pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES) + 10

        # override invalid position
        if pos > self.num_of_frames:
            pos = self.num_of_frames - 1

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos)

        # load the frame
        self.nextFrame()

    def JumpBackward(self):
        "Jumps back 10 frames"

        # check if there is a video opened
        if self.camera is None:
            return

        # get current position
        pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES) - 10

        # override invalid position
        if pos < 0:
            pos = 0

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos)

        # load the frame
        self.nextFrame()

    def JumpStart(self):
        """Jumps to the starting frame"""

        # check if there is a video opened
        if self.camera is None:
            return

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # load the frame
        self.nextFrame()

    def JumpEnd(self):
        """Jumps to the ending frame"""

        # check if there is a video opened
        if self.camera is None:
            return

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.num_of_frames - 1)

        # load the frame
        self.nextFrame()

    def positionVideo(self):
        """Positions the video according to the Slider value"""

        # check if there is a video opened
        if self.camera is None:
            return

        # load position from slider
        pos = self.VideoSLD.value()
        pos_in_frames = round(pos / 10000 * self.num_of_frames)

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos_in_frames)

        # load the frame
        self.nextFrame()

    def resizeEvent(self, e):
        "Handles the resizing of the window"
        # reloads the frame
        self.ReloadCurrentFrame()

        # execute parent functions
        return super().resizeEvent(e)

    def changeXoffset(self, delta):
        """Adjusts the X offset of the video displaying label"""

        # check if there is a video opened
        if self.camera is None:
            return

        # update the offset
        x_new = self.x_offset + delta

        # constrain with the boundaries
        if ceil(x_new + self.zoom * self.video_width / 2) >= floor(
            self.video_width / 2
        ):
            x_new = floor(self.video_width / 2 - self.zoom * self.video_width / 2)
        elif floor(self.video_width / 2) + x_new <= ceil(
            self.zoom * self.video_width / 2
        ):
            x_new = -floor(self.video_width / 2 - self.zoom * self.video_width / 2)
        self.x_offset = x_new

        # reload frame
        self.ReloadCurrentFrame()

    def changeYoffset(self, delta):
        """Adjusts the Y offset of the video displaying label"""

        # check if there is a video opened
        if self.camera is None:
            return

        # update offset
        y_new = self.y_offset + delta

        # constrain with the boundaries
        if ceil(y_new + self.zoom * self.video_height / 2) >= floor(
            self.video_height / 2
        ):
            y_new = floor(self.video_height / 2 - self.zoom * self.video_height / 2)

        elif floor(self.video_height / 2) + y_new <= ceil(
            self.zoom * self.video_height / 2
        ):
            y_new = -floor(self.video_height / 2 - self.zoom * self.video_height / 2)
        self.y_offset = y_new

        # reload frame
        self.ReloadCurrentFrame()

    def changeZoom(self, delta):
        """Zooms in or out of the frame, adjusts the offsets accordingly"""

        # chack is there is a video opened
        if self.camera is None:
            return

        # change the zoom
        z = self.zoom + delta
        if z <= 0.1:
            z = 0.1
        elif z >= 1:
            z = 1
        self.zoom = z

        # adjusting offset by constrains
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

        # reload frame
        self.ReloadCurrentFrame()

    def GridUpdate(self):
        """Updates the grid"""

        # constrain the input
        if self.GridGB.isChecked():
            if self.XGridLNE.text() == "" or int(self.XGridLNE.text()) == 0:
                self.XGridLNE.setText("1")
            if self.YGridLNE.text() == "" or int(self.YGridLNE.text()) == 0:
                self.YGridLNE.setText("1")

        # reload frame
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
        current_time = (index - 1) / self.fps
        h = int(current_time / 3600)
        min = int(current_time / 60) - 60 * h
        s = (current_time) - 60 * min
        return (
            str(h).zfill(2) + ":" + str(min).zfill(2) + ":" + str(f"{s:2.2f}").zfill(5)
        )

    def setresetStart(self):
        """Sets and resets the start of the tracked section"""

        # check if there is a video opened
        if self.camera is None:
            return

        # select frame
        if self.section_start is None:
            pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            if self.section_stop is not None:
                if self.section_stop == pos:
                    self.showWarningMessage(
                        "Cannot select the same strarting and ending frame!"
                    )
                    return
                if pos > self.section_stop:
                    self.showWarningMessage(
                        "The ending frame cannot be ahead of the starting frame!"
                    )
                    return
            self.section_start = pos
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

        # check if there is a video opened
        if self.camera is None:
            return

        # select frame
        if self.section_stop is None:
            pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            if self.section_start is not None:
                if self.section_start == pos:
                    self.showWarningMessage(
                        "Cannot select the same strarting and ending frame!"
                    )
                    return
                if pos < self.section_start:
                    self.showWarningMessage(
                        "The ending frame cannot be ahead of the starting frame!"
                    )
                    return
            self.section_stop = pos
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

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # reorganize the layout
        self.NewObjectBTN.setVisible(False)
        self.NameLNE.setVisible(True)
        self.PickPointBTN.setVisible(True)
        self.PickRectangleBTN.setVisible(True)
        self.SaveBTN.setVisible(True)
        self.CancelBTN.setVisible(True)
        self.VideoSLD.setEnabled(False)
        self.StartPauseBTN.setEnabled(False)
        self.ForwardBTN.setEnabled(False)
        self.BackwardBTN.setEnabled(False)

        # set position
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, (self.section_start - 1))
        self.nextFrame()
        self.ReloadCurrentFrame()

    def saveObject(self):
        """Saves temporary point and rectangle data in the objects_to_track list"""

        # get name from LineEdit or create one
        name = self.NameLNE.text()
        if name is None:
            name = f"obj-{len(self.objects_to_track)}"
        elif name == "":
            name = f"obj-{len(self.objects_to_track)}"
        if self.point_tmp is None:
            self.showWarningMessage("No point selected!")
            return
        if self.rect_tmp is None:
            self.showWarningMessage("No region selected!")
            return

        # Save object
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

        self.setFocus() # exits from lineedit

        # reload frame
        self.ReloadCurrentFrame()

    def pickPointStart(self):
        """Waits for a signal generated by VidLBL with the coordinates"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
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

        # disable region picking button
        self.PickRectangleBTN.setChecked(False)

        # save selected point
        self.VidLBL.press.connect(lambda x, y: self.savePoint(x, y))

    def pickRectStart(self):
        """Waits for a signal generated by VidLBL with coordinates"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
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

        # disable point selecting
        self.PickPointBTN.setChecked(False)

        # connect/disconnect signals
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

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # retrieve coordinates
        x, y = self.relative_to_cv(x, y)

        # store in temoporary variable
        self.rect_tmp = (x, y, x, y)

    def resizeRectangle(self, x, y):
        """Updates the rectangle when the mouse is moving"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # retrieve coordinates
        x1, y1 = self.relative_to_cv(x, y)
        x0, y0, _, _ = self.rect_tmp

        # update temporary variable
        self.rect_tmp = (x0, y0, x1, y1)

        # reload frame
        self.ReloadCurrentFrame()

    def saveRectangle(self, x, y):
        """Saves drawn rectangle"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # retrieve coordintes
        x1, y1 = self.relative_to_cv(x, y)
        x0, y0, _, _ = self.rect_tmp
        self.rect_tmp = (x0, y0, x1, y1)

        # disable select button
        self.PickRectangleBTN.setChecked(False)

        # jump to region selection
        self.pickRectStart()

        # reload frame
        self.ReloadCurrentFrame()

    def savePoint(self, x, y):
        """Saves the coordinates recieved from VidLBL"""

        # retrieve coordinates
        x, y = self.relative_to_cv(x, y)
        self.point_tmp = (x, y)

        # reload frame
        self.ReloadCurrentFrame()

        # disconnect signal and disable button
        self.VidLBL.press.disconnect()
        self.PickPointBTN.setChecked(False)

    def deleteObject(self, name):
        """Deletes the selected object from memory"""

        # get the index of the object
        index = next(
            (i for i, item in enumerate(self.objects_to_track) if item.name == name), -1
        )

        if index==-1:
            return

        # delete from list
        del self.objects_to_track[index]

        # delete from ListWidget
        i = self.ObjectLWG.takeItem(index)
        del i

        # delete from export dialog
        self.exportDialog.delete_object(name)

        # check if tracking is still available
        if (
            len(self.objects_to_track) == 0
            or self.section_start is None
            or self.section_stop is None
        ):
            self.TrackBTN.setEnabled(False)

        # reload frame
        self.ReloadCurrentFrame()

    def changeObjectDisplay(self, name):
        """Changes the visibility of the selected object"""

        # get object
        obj = get_from_list_by_name(self.objects_to_track, name)

        # change visibility
        if obj is not None:
            obj.visible = not obj.visible
            self.ReloadCurrentFrame()

    def setRuler(self):
        """Reorganises the GUI to make ruler control buttons visible connects mouse movement signals"""
        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # connect signals
        self.VidLBL.press.connect(lambda x, y: self.setRulerStart(x, y))
        self.VidLBL.moving.connect(lambda x, y: self.setRulerMove(x, y))

        # reorganize layout
        self.setRulerBTN.setVisible(False)
        self.saveRulerBTN.setVisible(True)
        self.removeRulerBTN.setVisible(True)
        self.mmLBL.setVisible(True)
        self.mmLNE.setVisible(True)

    def setRulerStart(self, x, y):
        """Captures the starting point of the ruler"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return

        # clear previous data
        self.ruler.clear()

        # retrieve coordinates
        x0, y0 = self.relative_to_cv(x, y)

        # set start point
        self.ruler.setP0(x0, y0)

        # reload frame
        self.ReloadCurrentFrame()

    def setRulerMove(self, x, y):
        """Refreshes the endpoint of the ruler"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # disable ready state
        self.ruler.rdy = False

        # retrieve coordinates
        x1, y1 = self.relative_to_cv(x, y)

        # set endpoint
        self.ruler.setP1(x1, y1)

        # reload frame
        self.ReloadCurrentFrame()

    def saveRuler(self):
        """Saves the ruler drawn by the user"""

        # read input value
        input = self.mmLNE.text()

        # check if ruler is selected properly
        if not self.ruler.displayable() or input == "":
            return

        # convert & calculate
        self.ruler.mm = float(input)
        self.ruler.calculate()

        # set ruler on export dialog
        self.exportDialog.setRuler(self.ruler.rdy)

        # reorganize layout
        self.saveRulerBTN.setVisible(False)
        self.changeRulerVisibilityBTN.setVisible(True)
        self.mmLNE.setVisible(False)
        self.mmLBL.setText(f"{self.ruler.mm_per_pix:.2f} mm/px")
        self.mmLBL.setAlignment(Qt.AlignCenter)

        # disconnect signals
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

        # reload frame
        self.ReloadCurrentFrame()

    def removeRuler(self):
        """Removes the ruler"""

        # reset ruler
        self.ruler.reset()

        # set ruler on export dialog
        self.exportDialog.setRuler(self.ruler.rdy)

        # reorganize layout
        self.setRulerBTN.setVisible(True)
        self.saveRulerBTN.setVisible(False)
        self.mmLBL.setVisible(False)
        self.mmLNE.setVisible(False)
        self.mmLBL.setText("mm")
        self.mmLNE.setText("")
        self.removeRulerBTN.setVisible(False)
        self.changeRulerVisibilityBTN.setVisible(False)

        # disconnect signals
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
        
        # reset focus
        self.setFocus()

        # reaload frame
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

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            return

        # connect signals
        self.VidLBL.press.connect(lambda x, y: self.setRoiStart(x, y))
        self.VidLBL.moving.connect(lambda x, y: self.setRoiMove(x, y))

        # reorganize layouts
        self.setRoiBTN.setVisible(False)
        self.saveRoiBTN.setVisible(True)
        self.delRoiBTN.setVisible(True)

    def setRoiStart(self, x, y):
        """Select the starting point of the Range Of Interest rectangle"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # retrieve and store
        x1, y1 = self.relative_to_cv(x, y)
        self.roi_rect = (x1, y1, x1, y1)

        # reload frame
        self.ReloadCurrentFrame()

    def setRoiMove(self, x, y):
        """Refreshes the endpoint of the Range Of Interest rectangle"""

        # check if video is opened, tracking section selected
        if (
            self.camera is None
            or self.section_start is None
            or self.section_stop is None
        ):
            self.showWarningMessage(
                "You must select the start and the end of the section to track!"
            )
            return

        # retrieve and store
        x1, y1 = self.relative_to_cv(x, y)
        self.roi_rect = (self.roi_rect[0], self.roi_rect[1], x1, y1)

        # rload frame
        self.ReloadCurrentFrame()

    def delRoi(self):
        """Removes the Range Of Interest rectangle"""

        # reset temporary data
        self.roi_rect = None

        # reorganize layout
        self.setRoiBTN.setVisible(True)
        self.saveRoiBTN.setVisible(False)
        self.delRoiBTN.setVisible(False)

        # disconnect signals
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

        # reload frame
        self.ReloadCurrentFrame()

    def saveRoi(self):
        """Saves the roi drawn by the user"""

        # hide button
        self.saveRoiBTN.setVisible(False)

        # disconnect signals
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

        # reload frame
        self.ReloadCurrentFrame()

    def showTrackingSettings(self):
        """Displays the dialog with the tracking settings"""

        # show tracking dialog
        if self.settingsDialog.exec_():

            # get tracker type
            tracker_type = self.settingsDialog.algoCMB.currentText()

            # get size change
            if self.settingsDialog.sizeCHB.isChecked():
                size = True
            else:
                size = False

            # get FPS
            if self.settingsDialog.fpsLNE.text() != "":
                fps = int(self.settingsDialog.fpsLNE.text())
            else:
                fps = self.fps

            # running the tracker
            self.runTracker(tracker_type, size, fps)

    def eventFilter(self, source, event):
        """Enables users to change X and Y offsets with the W-A-S-D butttons"""

        # chenge offset
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

        # call parent function
        return super(VideoWidget, self).eventFilter(source, event)

    def pix2mm(self, data):
        """Converts distance measured in pixels to m if the Ruler is set"""
        if self.ruler.mm_per_pix is not None:
            return data * self.ruler.mm_per_pix

    def pix2m(self, data):
        """Converts distance measured in pixels to m if the Ruler is set"""
        if self.ruler.mm_per_pix is not None:
            return data * self.ruler.mm_per_pix / 1000

    def runTracker(self, tracker_type, size, fps):
        """Runs the seleted tracking algorithm with the help of a QThread"""

        # resets the state in case of error
        self.mode = False

        # create tracking thread
        self.tracker = TrackingThreadV2(
            self.objects_to_track,
            self.camera,
            self.section_start,
            self.section_stop,
            tracker_type,
            size,
            fps,
            self.timestamp,
            self.roi_rect,
        )

        # connect signals ans start tracker
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
        """Shows the given error message in a new dialog"""
        msg = QMessageBox()
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/message.qss", "r") as style:
            msg.setStyleSheet(style.read())
        msg.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        msg.setWindowTitle("Error occured!")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    def showWarningMessage(self, message):
        """Shows the given warning message in a new dialog"""
        msg = QMessageBox()
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/message.qss", "r") as style:
            msg.setStyleSheet(style.read())
        msg.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        msg.setWindowTitle("Warning!")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    def trackingSucceeded(self):
        """After successful tracking change layout and functionality, open post-processing"""
        # cahnge mode
        self.mode = True
        self.reProcessBTN.setVisible(True)

        # reorganize layout
        self.TrackingSectionGB.setEnabled(False)
        self.ObjectsGB.setEnabled(False)
        self.exportVidBTN.setEnabled(True)
        self.resetAllBTN.setEnabled(True)
        if len(self.objects_to_track) >= 2:
            self.rotGB.setEnabled(True)
        self.playbackGB.setEnabled(True)
        self.roiGB.setEnabled(False)

        self.openPostProcessing()

    def openPostProcessing(self):
        # open post-processing settings
        if self.postProcessDialog.exec_():
            self.PostProcesser = PostProcesserThread(
                True,
                self.objects_to_track,
                self.timestamp[1] - self.timestamp[0],
                self.postProcessDialog.parameters,
            )
            self.PostProcesser.success.connect(self.calculationFinished)
            self.postProcessProgressDialog.updateName("Calculating derivatives...")
            self.postProcessProgressDialog.startAnimation()
            self.PostProcesser.error_occured.connect(self.showErrorMessage)
            self.postProcessProgressDialog.rejected.connect(self.PostProcesser.cancel)
            self.PostProcesser.finished.connect(self.postProcessProgressDialog.accept)
            self.PostProcesser.start()
            self.postProcessProgressDialog.show()

    def calculationFinished(self):
        """Helper function that's called if calculation ran successfully"""
        self.exportBTN.setEnabled(True)

    def resetAll(self):
        """Reset all data"""

        # reorganize layout and change mode
        self.mode = False
        self.reProcessBTN.setVisible(False)
        self.PropGB.setEnabled(True)
        self.TrackingSectionGB.setEnabled(True)
        self.ObjectsGB.setEnabled(True)
        self.ZoomControlGB.setEnabled(True)
        self.RulerGB.setEnabled(True)
        self.exportBTN.setEnabled(False)
        self.exportVidBTN.setEnabled(False)
        self.rotGB.setEnabled(False)
        self.roiGB.setEnabled(True)
        self.playbackGB.setEnabled(False)

        # reset properties
        self.section_start = 0  # to make the reset successful
        self.section_stop = 0
        self.setresetStart()
        self.setresetStop()

        # reset stored info
        self.objects_to_track = []
        self.rotations = []
        self.timestamp = []

        # clear temp data
        self.point_tmp = None
        self.rect_tmp = None  # x0, y0, x1, y1
        self.roi_rect = None

        # remove ruer cancel current object
        self.removeRuler()
        self.cancelObject()

        # clear objects
        self.ObjectLWG.clear()
        self.rotLWG.clear()
        self.exportDialog.delete_object("ALL")
        self.exportDialog.delete_rotation("ALL")

        # reload frame
        self.ReloadCurrentFrame()

    def saveRotation(self, rotation):
        """Helper function that saves rotation object"""
        self.rotations.append(rotation)

    def deleteRotation(self, name):
        # get the index of the object
        index = next(
            (i for i, item in enumerate(self.rotations) if item.__str__() == name), -1
        )

        if index==-1:
            return

        # delete from list
        del self.rotations[index]

        # delete from ListWidget
        i = self.rotLWG.takeItem(index)
        del i

        # delete from export dialog
        self.exportDialog.delete_rotation(name)



    def addRotation(self):
        """Add rotation object"""

        # check if there are 2 existing ibjects
        if len(self.objects_to_track) < 2:
            return

        # open settings dialog
        settings = RotationSettings()
        settings.set_params(self.objects_to_track)
        if settings.exec_():
            P1_name = settings.p1CMB.currentText()
            P2_name = settings.p2CMB.currentText()
            P1 = get_from_list_by_name(self.objects_to_track, P1_name)
            P2 = get_from_list_by_name(self.objects_to_track, P2_name)
            if P1 is not None and P2 is not None:
                # create rotation and run post-processing
                R = Rotation(P1, P2)
                R.calculate()
                if self.postProcessDialog.exec_():
                    self.PostProcesser = PostProcesserThread(
                        False,
                        R,
                        self.timestamp[1] - self.timestamp[0],
                        self.postProcessDialog.parameters,
                    )
                    self.PostProcesser.success.connect(self.calculationFinished)
                    self.progressDialog.updateName("Calculation in progress...")
                    self.progressDialog.updateBar(0)
                    self.PostProcesser.progressChanged.connect(
                        self.progressDialog.updateBar
                    )
                    self.PostProcesser.error_occured.connect(self.showErrorMessage)
                    self.progressDialog.rejected.connect(self.PostProcesser.cancel)
                    self.PostProcesser.finished.connect(self.progressDialog.accept)
                    self.PostProcesser.start()
                    self.progressDialog.show()
                    self.rotLWG.addItem(str(R))
                    self.rotations.append(R)
                    self.exportDialog.add_rotation(str(R))

        # delete setting dialog
        settings.deleteLater()

    def showExportDialog(self):
        """Show the export dialog"""
        self.exportDialog.setRuler(self.ruler.rdy)
        self.exportDialog.show(
            self.settingsDialog.sizeCHB.isChecked(),
            all(x.can_plot() for x in self.objects_to_track),
            all(x.can_plot() for x in self.rotations),
        )

    def getPlotData(self, parameters):
        """Get data collected by the Export dialog and creates the plot or file"""

        # check mode
        if not self.mode:
            return

        # variables for export
        exp_ok = False
        data = None

        # movement
        if parameters["mode"] == "MOV" and len(parameters["objects"]):
            if parameters["ax"] == "BOTH":
                data = np.zeros(
                    (len(self.timestamp), 2 * len(parameters["objects"]) + 1)
                )
            else:
                data = np.zeros((len(self.timestamp), len(parameters["objects"]) + 1))
            cols = []

            # get timestamp
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")
            for i in range(len(parameters["objects"])):

                # timestamp only needed once
                obj_name = parameters["objects"][i]
                obj = get_from_list_by_name(self.objects_to_track, obj_name)
                if obj is None:
                    return

                # position
                if parameters["prop"] == "POS":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.position[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = -obj.position[:, 1]
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    elif parameters["ax"] == "BOTH":
                        data[:, i*2 + 1] = obj.position[:, 0]
                        cols.append(obj.name + " - X")
                        data[:, i*2 + 2] = -obj.position[:, 1]
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Position"

                elif parameters["prop"] == "VEL":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.velocity[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = -obj.velocity[:, 1] 
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    elif parameters["ax"] == "BOTH":
                        data[:, i*2 + 1] = obj.velocity[:, 0]
                        cols.append(obj.name + " - X")
                        data[:, i*2 + 2] = -obj.velocity[:, 1]
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Velocity"

                elif parameters["prop"] == "ACC":

                    # axis
                    if parameters["ax"] == "XT":
                        data[:, i + 1] = obj.acceleration[:, 0]
                        cols.append(obj.name + " - X")
                        exp_ok = True
                    elif parameters["ax"] == "YT":
                        data[:, i + 1] = -obj.acceleration[:, 1]
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    elif parameters["ax"] == "BOTH":
                        data[:, i*2 + 1] = obj.acceleration[:, 0]
                        cols.append(obj.name + " - X")
                        data[:, i*2 + 2] = -obj.acceleration[:, 1] 
                        cols.append(obj.name + " - Y")
                        exp_ok = True
                    title = "Acceletration"

            if parameters["unit"] == "mm":
                data = pix2mm(data, self.ruler.mm_per_pix)
            elif parameters["unit"] == "m":
                data = pix2m(data, self.ruler.mm_per_pix)

        # rotation
        elif parameters["mode"] == "ROT":
            data = np.zeros((len(self.timestamp), len(parameters["rotations"]) + 1,))
            cols = []
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")

            for i in range(len(parameters["rotations"])):

                rot_name = parameters["rotations"][i]
                rot = get_from_list_by_name(self.rotations, rot_name)
                if rot is None:
                    return

                if parameters["prop"] == "POS":
                    data[:, i + 1] = rot.rotation
                    cols.append(rot.P1.name + " + " + rot.P2.name + " rotation")
                    title = "Rotation"
                    exp_ok = True
                elif parameters["prop"] == "VEL":
                    data[:, i + 1] = rot.ang_velocity
                    cols.append(rot.P1.name + " + " + rot.P2.name + " angular velocity")
                    title = "Angular velocity"
                    exp_ok = True

                elif parameters["prop"] == "ACC":
                    data[:, i + 1] = rot.ang_acceleration
                    cols.append(
                        rot.P1.name + " + " + rot.P2.name + " angular acceleration"
                    )
                    title = "Angular acceleration"
                    exp_ok = True

            if parameters["unit"] == "DEG":
                data = rad2deg_(data)

        # size change
        elif parameters["mode"] == "SIZ":
            data = np.zeros((len(self.timestamp), len(parameters["objects"]) + 1,))
            cols = []
            data[:, 0] = np.asarray(self.timestamp)
            cols.append("Time (s)")
            for i in range(len(parameters["objects"])):

                obj_name = parameters["objects"][i]
                obj = get_from_list_by_name(self.objects_to_track, obj_name)
                if obj is None:
                    return

                # position
                data[:, i + 1] = np.asarray(obj.size_change)
                cols.append(obj.name + " size change")
                title = "Size change"
                exp_ok = True

        # chack if data collection was successfull
        if data is None or not exp_ok:
            return

        # plot data
        if parameters["out"] == "PLOT":
            df = pd.DataFrame(data, columns=cols)
            self.plotter = PlotDialog(df, title, get_unit(parameters))
            self.plotter.show()

        # export data
        elif parameters["out"] == "EXP":
            unit = get_unit_readable(parameters)
            cols = [f"{item} ({unit})" for item in cols]
            cols[0]="Time (s)"
            df = pd.DataFrame(data, columns=cols)
            save_name = QFileDialog.getSaveFileName(
                self,
                "Save data",
                "/",
                "CSV file (*.csv);;Text file (*.txt);;Excel file (*.xlsx)",
            )
            if save_name[0] != "":
                if save_name[1] == "Excel file (*.xlsx)":
                    df.to_excel(save_name[0])
                else:
                    df.to_csv(save_name[0])

    def exportVideo(self):
        """Exports the video with the tracked objects in MP! format"""

        # get filename
        save_name = QFileDialog.getSaveFileName(
            self, "Export Video", "/", "MP4 file (*.mp4)",
        )

        # save video file
        if save_name[0] != "":
            # create thread, connect signals
            self.exporter = ExportingThread(
                self.camera,
                self.objects_to_track,
                self.section_start,
                self.section_stop,
                save_name[0],
                self.fps,
                self.boxCHB.isChecked(),
                self.pointCHB.isChecked(),
                round(self.playbackSLD.value() * self.num_of_frames / 100),
            )
            self.progressDialog.updateName("Exporting video to " + save_name[0])
            self.progressDialog.rejected.connect(self.exporter.cancel)
            self.exporter.progressChanged.connect(self.progressDialog.updateBar)
            self.exporter.finished.connect(self.progressDialog.accept)
            self.exporter.start()
            self.progressDialog.show()