# -*- coding: utf-8 -*-
"""
@author: Jardi
"""

from __future__ import print_function, division

import sys
sys.path.insert(1, 'GUI')

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QFrame
import dialog
from application import Window


if __name__ == "__main__":
    ## Start an application
    app = QApplication(sys.argv)
    
    ## Quit application when the window is closed
    app.lastWindowClosed.connect(app.quit)
    
    ## Run the application
    application = Window(app)
    application.show()
    
    sys.exit(app.exec())