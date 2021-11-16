import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpacerItem,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QMainWindow,
    QGridLayout,
    QFileDialog,
    QGroupBox,
)
import cv2
from classes import ObjectToTrack, VideoLabel  # TimeInputDialog
from math import floor, ceil


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()
        self.camera = None
        self.fps = None
        self.timer = QTimer()
        self.num_of_frames = 0
        self.x_offset = 0
        self.y_offset = 0
        self.zoom = 1  # from 1 goes down to 0
        self.section_start = None
        self.section_stop = None
        self.filename = ""
        self.video_width = None
        self.video_height = None
        self.objects_to_track = []
        self.point_tmp = None
        self.rect_tmp = None  # x0,y0,x1,y1
        self.timer.setTimerType(Qt.PreciseTimer)
        self.setWindowTitle("VideoWidget")
        self.setGeometry(100, 100, 1280, 720)  # x,y,w,h
        self.initUI()
        self.show()

    def initUI(self):
        """Creating and setting up the user interface"""
        self.showMaximized()

        # Video open
        OpenBTN = QPushButton("Open Video")
        OpenBTN.clicked.connect(self.openNewFile)

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
        TrackingSectionGB = QGroupBox()
        TrackingSectionGB.setTitle("Section to track")
        TrackingSectionGB.setLayout(TrackingSectionLayout)
        TrackingSectionGB.setFixedWidth(200)

        # Adding points to track
        self.NewObjectBTN = QPushButton()
        self.NewObjectBTN.setText("Add new point")
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

        self.PickLayout = QHBoxLayout()
        self.PickLayout.addWidget(self.PickPointBTN)
        self.PickLayout.addWidget(self.PickRectangleBTN)

        self.ObjectsLayout = QVBoxLayout()
        self.ObjectsLayout.addWidget(self.NewObjectBTN)
        self.ObjectsLayout.addWidget(self.NameLNE)
        self.ObjectsLayout.addLayout(self.PickLayout)
        self.ObjectsLayout.addWidget(self.SaveBTN)

        ObjectsGB = QGroupBox()
        ObjectsGB.setTitle("Objects to track")
        ObjectsGB.setLayout(self.ObjectsLayout)
        ObjectsGB.setFixedWidth(200)

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

        ForwardBTN = QPushButton()
        ForwardBTN.setIcon(QIcon("images/forward.svg"))
        ForwardBTN.setMinimumSize(QSize(40, 40))
        ForwardBTN.clicked.connect(self.JumpForward)

        BackwardBTN = QPushButton()
        BackwardBTN.setIcon(QIcon("images/backward.svg"))
        BackwardBTN.setMinimumSize(QSize(40, 40))
        BackwardBTN.clicked.connect(self.JumpBackward)

        self.VideoSLD = QSlider(Qt.Horizontal)
        self.VideoSLD.setMinimum(0)
        self.VideoSLD.setMaximum(10000)
        self.VideoSLD.setValue(0)
        self.VideoSLD.valueChanged.connect(self.positionVideo)

        # Add controllers to layout
        PlayerControlLayout = QHBoxLayout()
        PlayerControlLayout.addWidget(self.StartPauseBTN)
        PlayerControlLayout.addWidget(StopBTN)
        PlayerControlLayout.addWidget(BackwardBTN)
        PlayerControlLayout.addWidget(ForwardBTN)
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
        self.VidLBL.release.connect(lambda x_rel, y_rel: self.drawPoint(x_rel, y_rel))

        # Label for timestap
        self.VidTimeLBL = QLabel()
        self.VidTimeLBL.setText("-/-")
        self.VidTimeLBL.setAlignment(Qt.AlignCenter)
        self.VidTimeLBL.setStyleSheet("font-weight: bold; font-size: 14px")

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

        ZoomControlGB = QGroupBox()
        ZoomControlGB.setTitle("Zoom control")
        ZoomControlGB.setLayout(ZoomControlLayout)

        # SideLayout
        SideLayout = QVBoxLayout()
        SideLayout.addWidget(OpenBTN)
        SideLayout.addWidget(self.PropGB)
        SideLayout.addWidget(TrackingSectionGB)
        SideLayout.addWidget(ObjectsGB)
        SideLayout.addWidget(ZoomControlGB)
        SideLayout.addItem(
            QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        SideLayout.addWidget(self.VidTimeLBL)
        SideLayout.setContentsMargins(0, 0, 0, 12)

        # Vertical layout to wrap the player
        VideoControlLayout = QVBoxLayout()
        VideoControlLayout.addWidget(self.VidLBL)
        VideoControlLayout.addLayout(PlayerControlLayout)

        # Horizontal layout with zoom added
        PlayerLayout = QHBoxLayout()
        PlayerLayout.addLayout(VideoControlLayout)
        PlayerLayout.addLayout(SideLayout)

        # Overall layout
        self.setLayout(PlayerLayout)

    def openNewFile(self):
        """Opens a new video file"""
        filename = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "./",
            "MP4 file (*.mp4);;MOV file (*.mov);;AVI file (*.avi)",
        )[0]
        if filename != "":
            self.filename = filename
            self.openVideo()

    def openVideo(self):
        """Creates the VideoCapture object and gets required properties"""
        self.camera = cv2.VideoCapture(self.filename)

        # important properties
        self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        self.num_of_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # display properties
        self.FileNameLBL.setText(f"Filename: {QFileInfo(self.filename).fileName()}")
        self.ResolutionLBL.setText(
            f"Resolution: {self.video_width}x{self.video_height}"
        )
        self.FpsLBL.setText(f"FPS: {self.fps}")
        self.LengthLBL.setText(
            f"Video length: {self.time_to_display(self.num_of_frames)}"
        )
        self.PropGB.setVisible(True)

        self.timer.timeout.connect(self.nextFrame)
        self.nextFrame()

    def closeVideo(self):
        """Closes the video, releases the cam"""
        self.camera.release()
        self.PropGB.setVisible(False)
        self.VidTimeLBL.setText("-/-")
        self.VidLBL.setPixmap(QPixmap("images/video.svg"))

    def StartPauseVideo(self):
        """Starts and pauses the video"""
        if self.camera is None:
            self.StartPauseBTN.setChecked(False)
            return
        if self.StartPauseBTN.isChecked():
            if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.num_of_frames:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.timer.start(int(1000 / self.fps))
            self.StartPauseBTN.setIcon(QIcon("images/pause.svg"))
        else:
            self.timer.stop()
            self.StartPauseBTN.setIcon(QIcon("images/play.svg"))

    def nextFrame(self):
        """Loads and displays the next frame from the video feed"""
        ret, frame = self.camera.read()
        if ret:
            frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
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
            slider_pos = int(pos_in_frames / self.num_of_frames * 10000)
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
                if True:
                    # previous
                    for obj in self.objects_to_track:
                        x, y = obj.point
                        frame = cv2.drawMarker(
                            frame, (x, y), (0, 0, 255), 0, thickness=2
                        )
                        x0, y0, x1, y1 = obj.rectangle
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
                    # current
                    if self.point_tmp is not None:
                        x, y = self.point_tmp
                        frame = cv2.drawMarker(
                            frame, (x, y), (0, 0, 255), 0, thickness=2
                        )
                    if self.rect_tmp is not None:
                        x0, y0, x1, y1 = self.rect_tmp
                        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                frame = crop_frame(frame, self.x_offset, self.y_offset, self.zoom)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(
                    frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
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
        pos_in_frames = int(pos / 10000 * self.num_of_frames)
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, pos_in_frames)
        self.nextFrame()

    def resizeEvent(self, e):
        self.ReloadCurrentFrame()
        return super().resizeEvent(e)

    def changeXoffset(self, delta):
        """Adjusts the X offset"""
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
        print(f"xoffset:{self.x_offset} yoffset:{self.y_offset}")
        self.ReloadCurrentFrame()

    def changeYoffset(self, delta):
        """Adjusts the Y offset"""
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
        print(f"x:{x} y:{y}")
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

    def addNewObject(self):
        """Cunfigures the GUI for setting the point and tectangle that will be used for the tracking"""
        if self.camera is None or self.section_start is None:
            return

        # reorganize the layout
        self.NewObjectBTN.setVisible(False)
        self.NameLNE.setVisible(True)
        self.PickPointBTN.setVisible(True)
        self.PickRectangleBTN.setVisible(True)
        self.SaveBTN.setVisible(True)

        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)
        self.nextFrame()
        self.VideoSLD.setEnabled(False)
        self.StartPauseBTN.setEnabled(False)

    def saveObject(self):
        """Saves temporary point and rectangle data in the objects_to_track list"""
        name = self.NameLNE.text()
        if self.point_tmp is None or self.rect_tmp is None or name == "":
            return
        # save object
        O = ObjectToTrack(name, self.point_tmp, self.rect_tmp)
        self.objects_to_track.append(O)

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
        self.NameLNE.setVisible(False)
        self.NameLNE.clear()
        self.PickPointBTN.setVisible(False)
        self.PickRectangleBTN.setVisible(False)
        self.SaveBTN.setVisible(False)
        self.NewObjectBTN.setVisible(True)

        # reorganize layout
        self.ReloadCurrentFrame()

    def pickPointStart(self):
        """Waits for a signal generated by VidLBL with the coordinates"""
        if self.camera is None or self.section_start is None:
            return
        self.VidLBL.press.connect(lambda x, y: self.savePoint(x, y))

    def pickRectStart(self):
        """Waits for a signal generated by VidLBL with coordinates"""
        if self.camera is None or self.section_start is None:
            return
        if self.PickRectangleBTN.isChecked():
            self.VidLBL.moving.connect(lambda x, y: self.resizeRectangle(x, y))
            self.VidLBL.press.connect(lambda x, y: self.initRectangle(x, y))
        else:
            try:
                self.VidLBL.moving.disconnect()
            except:
                pass
            try:
                self.VidLBL.press.disconnect()
            except:
                pass

    def initRectangle(self, x, y):
        """Initializes the rectangle when the mouse button is pressed"""
        if self.camera is None or self.section_start is None:
            return
        x, y = self.relative_to_cv(x, y)
        self.rect_tmp = (x, y, x, y)

    def resizeRectangle(self, x, y):
        """Updates the rectangle when the mosue is moving"""
        if self.camera is None or self.section_start is None:
            return
        x1, y1 = self.relative_to_cv(x, y)
        x0, y0, _, _ = self.rect_tmp
        self.rect_tmp = (x0, y0, x1, y1)
        self.ReloadCurrentFrame()

    def savePoint(self, x, y):
        """Saves the coordinates recieved from VidLBL"""
        x, y = self.relative_to_cv(x, y)
        self.point_tmp = (x, y)
        print(self.point_tmp)
        self.ReloadCurrentFrame()
        self.VidLBL.press.disconnect()
        self.PickPointBTN.setChecked(False)


def crop_frame(frame, x_offset, y_offset, zoom):
    """Crops the frame according to offset and zoom parameters"""
    x0 = ceil(frame.shape[1] / 2)
    y0 = ceil(frame.shape[0] / 2)
    return frame[
        (y0 + y_offset - round(y0 * zoom)) : (y0 + y_offset + round(y0 * zoom)),
        (x0 + x_offset - round(x0 * zoom)) : (x0 + x_offset + round(x0 * zoom)),
    ]


if __name__ == "__main__":
    App = QApplication(sys.argv)
    root = VideoWidget()
    sys.exit(App.exec())