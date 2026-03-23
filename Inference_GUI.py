#  MWD-Inference-GUI
#  --------------------------------------------------------------------------------------------------------

#  Authors:
#      Robert J. Elliott, py19rje@leeds.ac.uk
#      Daniel J. Read, d.j.read@leeds.ac.uk
#      Luisa Cutillo, l.cutillo@leeds.ac.uk
#      Johan Mattsson, k.j.l.mattsson@leeds.ac.uk

#  GitHub:
#      https://github.com/py19rje/MWD-Inference-GUI

#  --------------------------------------------------------------------------------------------------------

#  Copyright (2026): Robert J. Elliott, University of Leeds

#  This file is part of the software MWD-Inference-GUI

#  MWD-Inference-GUI is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  MWD-Inference-GUI is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this software.  If not, see <http://www.gnu.org/licenses/>.

#  This work forms part of the research programme of DPI, project \#861.

#  --------------------------------------------------------------------------------------------------------

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

    pathToGIF = "graphics/splash_with_logo.gif"
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
     