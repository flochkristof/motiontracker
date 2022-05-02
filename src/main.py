import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from gui import VideoWidget
from PyQt5.QtGui import QPixmap


# run the application
if __name__ == "__main__":
    # create app
    App = QApplication(sys.argv)

    # splash screen for loading
    splash = QSplashScreen(QPixmap("images/logo.svg"))
    splash.show()
    App.processEvents()

    # open application
    root = VideoWidget()
    root.show()

    # close splash
    splash.finish(root)
    sys.exit(App.exec())
