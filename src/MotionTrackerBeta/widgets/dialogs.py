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

from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSpacerItem,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QLineEdit,
    QWidget,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QFrame,
)
from PyQt5.QtGui import (
    QMouseEvent,
    QCursor,
    QWheelEvent,
    QIntValidator,
    QIcon,
    QDoubleValidator,
    QMovie,
)
from PyQt5.QtCore import Qt, pyqtSignal
from MotionTrackerBeta.functions.helper import *
from MotionTrackerBeta.classes.classes import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import os


MODE=True # disables/ enables the optimization based differentiation methods
# True: python environment
# False: complied exe


class PostProcessSettings(QDialog):
    """Modal dialog for users to specify post processing settings"""

    # create signal
    diffdata = pyqtSignal(tuple)

    def __init__(self, parent=None):

        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Post-processing settings")
        self.setModal(True)
        self.setObjectName("postprocess_window")
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/postprocess.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
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
                "Linear Models",
            ]
        )
        if MODE:
            self.diffFamilyCMB.addItem("Total Variation Regularization")

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
        self.iterationsLNE.setValidator(QIntValidator(0, 1000))
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

        # default setting is the manually specified
        self.manSpecRDB.setChecked(True)

        if not MODE:
            self.optRDB.setEnabled(False)
            note=QLabel("The hyperparameter optimization is not available in the complied software. Run the source code in a Python enviroment to access this feature!")
            note.setWordWrap(True)
            note.setObjectName("note")
            self.cutoffOptGB.setVisible(False)
            optLayout.addWidget(note)

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
                    "Convex Total Variation Regularization with Regularized Velocity",
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
        """Changes the accessibility of input fields based on selected mode"""

        # manually specified input params
        if self.manSpecRDB.isChecked():
            for p in self.parameters_list:
                p.setEnabled(True)
            self.cutoffOptGB.setEnabled(False)

        # optimization based input data
        elif self.optRDB.isChecked():
            if MODE:
                self.cutoffOptGB.setEnabled(True)
                for p in self.parameters_list:
                    p.setEnabled(False)

    def diffSpecUpdated(self):
        """Reorganize input fields based on selected algorithm"""

        # get algorithm
        algo = self.diffSpecCMB.currentText()

        # reorganize
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
        if not MODE:
            self.cutoffOptGB.setVisible(False)

    def collectParameters(self):
        """Collects the needed parameters of the selected differentiation algoritms"""

        # algorithm without params
        algo = self.diffSpecCMB.currentText()
        if algo in {"First Order Finite Difference", "Second Order Finite Difference"}:
            return (False, algo, None)

        # optimization based algorithms, only one input parameter
        elif self.optRDB.isChecked():
            if self.cutoffOptLNE.text() != "":
                return (True, algo, float(self.cutoffOptLNE.text()))
            else:
                return None

        # manually specified parameters
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
                print("button pressed")
                if self.regParLNE.text != "":
                    return (
                        False,
                        algo,
                        [float(self.regParLNE.text())],
                        {"solver": "CVXOPT"},
                    )
                else:
                    print("regpar invalid")
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
        """Initialization"""

        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Tracking settings")
        self.setModal(True)
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/tracking.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        self.setObjectName("tracker_window")
        self.setFixedSize(270, 200)

        ## initilaize and organize layout

        # tracking algorithm
        algoLBL = QLabel("Tracking Algorithm:")
        self.algoCMB = QComboBox()
        self.algoCMB.addItems(
            ["CSRT", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE"]
        )

        # size change CheckBox
        sizeLBL = QLabel("Track the size change of objects:")
        self.sizeCHB = QCheckBox()
        self.sizeCHB.setLayoutDirection(Qt.RightToLeft)
        self.sizeCHB.stateChanged.connect(self.sizeMode)

        # FPS input LineEdit
        fpsLBL = QLabel("Real FPS:")
        self.fpsLNE = QLineEdit()
        self.fpsLNE.setValidator(QIntValidator(1, 100000))
        self.notificationLBL = QLabel(
            "Important information: Only the CSRF algorithm is capable of tracking the size change of an object!"
        )

        # notification LBL
        self.notificationLBL.setVisible(False)
        self.notificationLBL.setWordWrap(True)

        # button
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


class RotationSettings(QDialog):
    """Dialog to select points for rotation tracking"""

    def __init__(self, parent=None):
        """Initialization"""

        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Rotation settings")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        self.setObjectName("rotation")
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/rotation.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setModal(True)
        # self.setAttribute(Qt.WA_DeleteOnClose) TODO

        # initialize and organize layout
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


class TrackingProgress(QDialog):
    """A modal dialog that shows the progress of the tracking"""

    def __init__(self, parent=None):
        """Initialization"""

        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Calculation in progress...")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/progress.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setObjectName("progress")
        self.setModal(True)

        # initialize and organize layout
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
        """Updates Label text"""
        self.label.setText(name)

    def updateBar(self, value):
        """Updates the ProgressBar"""
        self.progressbar.setValue(value)


class CalculationProgress(QDialog):
    """A modal dialog that shows the progress of the tracking"""

    def __init__(self, parent=None):
        """Initialization"""

        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Calculation in progress...")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/progress.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setObjectName("progress_spinner")
        self.setModal(True)

        # initialize and organize layout
        Layout = QVBoxLayout()
        vLayout = QHBoxLayout()
        self.label = QLabel()
        self.movieLBL = QLabel()
        self.movieLBL.setScaledContents(True)
        self.movieLBL.setMaximumHeight(50)
        self.movieLBL.setMaximumWidth(50)
        self.label.setStyleSheet("text-align: center;")
        cancelProgressBTN = QPushButton("Cancel")
        cancelProgressBTN.clicked.connect(self.rejected)
        self.movie = QMovie(os.path.dirname(os.path.dirname(__file__))+"/images/loader.gif")
        self.movieLBL.setMovie(self.movie)
        self.movieLBL.setStyleSheet("text-align: center;")
        vLayout.addWidget(self.label, Qt.AlignCenter)
        vLayout.addWidget(self.movieLBL, Qt.AlignCenter)
        Layout.addLayout(vLayout)
        Layout.addWidget(cancelProgressBTN)
        self.setLayout(Layout)

    def updateName(self, name):
        """Updates Label text"""
        self.label.setText(name)

    def startAnimation(self):
        self.movie.start()

    def accept(self):
        self.movie.stop()
        return super().accept()


class ExportDialog(QDialog):
    """Handles the exporting of the data collected by the trackers"""

    export = pyqtSignal(dict)

    def __init__(self, parent=None):
        # call parent function
        super().__init__(parent)

        # styling
        self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        self.setObjectName("export_dialog")
        with open(os.path.dirname(os.path.dirname(__file__))+"/style/export.qss", "r") as style:
            self.setStyleSheet(style.read())
        self.setWindowTitle("Export options")
        self.setModal(False)

        # list of objects and data initialization
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
        """Shows the widget"""
        self.manage_rotation(rot)
        self.manage_size(size)
        self.manage_mov(mov)
        return super().show()

    def setRuler(self, rdy):
        """Sets the ruler properties"""
        if rdy:
            self.mmRDB.setEnabled(True)
            self.mRDB.setEnabled(True)
        else:
            self.mmRDB.setEnabled(False)
            self.mRDB.setEnabled(False)

    def manage_rotation(self, rot):
        """Manages widget properties based on available data"""
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
        """Managee widget properties based on available data"""
        if size != 0:
            self.sizeFrame.setHidden(False)
        else:
            self.sizeFrame.setHidden(True)

    def manage_mov(self, mov):
        """Manages widget properties based on available data"""
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
        """Add object CheckBox"""
        checkbox1 = QCheckBox(object_name)
        self.objNameLayout.addWidget(checkbox1)
        self.obj_checkboxes.append(checkbox1)
        checkbox2 = QCheckBox(object_name)
        self.sizeObjLayout.addWidget(checkbox2)
        self.size_checkboxes.append(checkbox2)

    def delete_object(self, object_name):
        """Deletes object CheckBox"""

        # all checkboxes
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

        # delete from size change
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
        """Adds CheckBox with rotation name"""

        checkbox1 = QCheckBox(rot_name)
        self.rotNameLayout.addWidget(checkbox1)
        self.rot_checkboxes.append(checkbox1)

    def delete_rotation(self, rot_name):
        """Deletes rotation CheckBox"""

        # clear all rotation
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

        # delete from size change
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
        """Collects movement data properties and emits signal"""

        # set default params
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "MOV"

        # get objects
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()

        # check if params are collected properly
        if all(
            k in self.parameters
            for k in ("out", "mode", "objects", "prop", "ax", "unit")
        ):
            # emit signal
            self.export.emit(self.parameters)

    def export_movement(self):
        """Collects movement data properties and emits signal"""

        # set default params
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "MOV"
        objs = [
            checkbox.text() for checkbox in self.obj_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs
        self.get_movement_parameters()

        # check if params are collected properly
        if all(
            k in self.parameters
            for k in ("out", "mode", "objects", "prop", "ax", "unit")
        ):
            # emit signal
            self.export.emit(self.parameters)

    def plot_rotation(self):
        """Collect rotation data properties and emits signal"""

        # set default params
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "ROT"

        # get rotation objects
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots
        self.get_rotation_parameters()

        # check if parameters are collected properly
        if all(
            k in self.parameters for k in ("out", "mode", "rotations", "prop", "unit")
        ):
            # emit signal
            self.export.emit(self.parameters)

    def export_rotation(self):
        """Collects rotation data properties and emits signal"""

        # set default params
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "ROT"

        # get rotations objects
        rots = [
            checkbox.text() for checkbox in self.rot_checkboxes if checkbox.isChecked()
        ]
        self.parameters["rotations"] = rots

        # get parameters
        self.get_rotation_parameters()

        # check if parameters are collected properly
        if all(
            k in self.parameters for k in ("out", "mode", "rotations", "prop", "unit")
        ):
            # emit signal
            self.export.emit(self.parameters)

    def plot_size(self):
        """Collects size data properties and emits signal"""

        # get parameters
        self.parameters.clear()
        self.parameters["out"] = "PLOT"
        self.parameters["mode"] = "SIZ"

        # get objects
        objs = [
            checkbox.text() for checkbox in self.size_checkboxes if checkbox.isChecked()
        ]
        self.parameters["objects"] = objs

        # emit signals
        self.export.emit(self.parameters)

    def export_size(self):
        """Collects size data properties and emits signal"""

        # get parameters
        self.parameters.clear()
        self.parameters["out"] = "EXP"
        self.parameters["mode"] = "SIZ"
        objs = [
            checkbox.text() for checkbox in self.size_checkboxes if checkbox.isChecked()
        ]

        # get objects
        self.parameters["objects"] = objs

        # emit signals
        self.export.emit(self.parameters)

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
        # get PROPERTY
        if self.rot_posRDB.isChecked():
            self.parameters["prop"] = "POS"
        elif self.rot_velRDB.isChecked():
            self.parameters["prop"] = "VEL"
        elif self.rot_accRDB.isChecked():
            self.parameters["prop"] = "ACC"
        else:
            self.parameters.clear()

        # get UNIT
        if self.degRDB.isChecked():
            self.parameters["unit"] = "DEG"
        elif self.radRDB.isChecked():
            self.parameters["unit"] = "RAD"
        else:
            self.parameters.clear()


class PlotDialog(QWidget):
    def __init__(self, df, title, unit, parent=None):
        # parent function
        super().__init__(parent)

        # styling
        self.setWindowTitle("Figure")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.dirname(__file__))+"/images/logo.svg"))
        self.setAttribute(Qt.WA_DeleteOnClose)

        # create figure
        self.figure = Figure()
        ax = self.figure.add_subplot(111)
        ax.clear()

        # plot data
        df.plot(kind="line", x=0, ax=ax)
        ax.set_ylabel(unit)
        ax.set_title(title)

        # draw to qt
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.draw()

        # toolbar for options
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # organize layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)