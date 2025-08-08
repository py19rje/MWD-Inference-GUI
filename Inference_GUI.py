import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QSplashScreen)
from PyQt5.QtGui import QIcon, QMovie, QPixmap, QPainter
from PyQt5.QtCore import Qt, QTimer
import os
import ctypes
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
           
def set_app_icon(app):
    icon_path = "graphics/NN.ico" 

    if sys.platform == "win32":
        myappid = "Inference_GUI_app"  
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(myappid))

    elif sys.platform == "darwin":  
        icon_path = "graphics/NN.icns"  

    if os.path.exists(icon_path):  
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Warning: Icon file not found at {icon_path}")
        return

class MovieSplashScreen(QSplashScreen):
    def __init__(self, pathToGIF):
        self.movie = QMovie(pathToGIF)
        self.movie.jumpToFrame(0)
        pixmap = QPixmap(self.movie.frameRect().size())
        QSplashScreen.__init__(self, pixmap)
        self.movie.frameChanged.connect(self.repaint)

    def showEvent(self, event):
        self.movie.start()

    def hideEvent(self, event):
        self.movie.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)
        
def launch_inference_GUI():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    pathToGIF = "graphics/splash.gif"
    splash = MovieSplashScreen(pathToGIF)
    splash.show()

    def initialize_main_window():
        set_app_icon(app)
        from modules.MainWindow import MainWindow  
        window = MainWindow()
        splash.close()
        window.show()

    QTimer.singleShot(1500, initialize_main_window)  # Delay by 100ms to ensure the splash screen is fully displayed
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    launch_inference_GUI()
     