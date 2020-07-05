# -*- coding: utf-8 -*-
"""
@author: Jardi
"""

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QFrame
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
import torch
from torchvision import transforms, models
from PIL import Image
from dialog import Ui_AFMNet  # importing our generated file
import nets
from Dataclass import Dataclass
import sys
import os
import pickle

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    text = pyqtSignal(str)
    image = pyqtSignal(QPixmap)
    
    def __init__(self, dirPath, device):
        super().__init__()
        self.dirPath = dirPath
        self.device = device
    
    @pyqtSlot(str, torch.nn.Module)
    def loadModel(self, modelPath, net):
        self.net = net
        self.progress.emit(25)
        
        try:
            self.mean, self.std = nets.load(net, modelPath, device = self.device)
            self.net.to(self.device)
        except RuntimeError:
            self.text.emit("This model does not have a " + type(self.net).__name__ + " architecture! Choose again!")
            self.finished.emit()
            return
        except (FileNotFoundError, pickle.UnpicklingError):
            self.text.emit("File not supported!")
            self.finished.emit()
            return
        
        self.progress.emit(70)
        
        if hasattr(self, "fileName"):
            self.text.emit("You can now use the model to predict whether the AFM image is good or bad")
        else:
            self.text.emit("Upload an AFM image!")
        
        self.finished.emit()
    
    @pyqtSlot(str)
    def uploadImage(self, fileName):
        self.fileName = fileName
        
        self.progress.emit(30)
        if not (self.fileName.endswith('.bmp') or self.fileName.endswith('.gif') or self.fileName.endswith('.jpg') 
                or self.fileName.endswith('.jpeg') or self.fileName.endswith('.png') or self.fileName.endswith('.pbm')
                or self.fileName.endswith('.tiff') or self.fileName.endswith('.svg') or self.fileName.endswith('.xbm')):
            self.text.emit("File not supported!")
            self.finished.emit()
            return
        
        self.progress.emit(50)
        
        pixmap = QPixmap(self.fileName)
        self.progress.emit(80)
        self.image.emit(pixmap)
        if not hasattr(self, "net"):
            self.text.emit("Select a network architecture and load a pretrained model.")
        else:
            self.text.emit("You can now use the model to predict whether the AFM image is good or bad!")

        self.progress.emit(100)
        self.finished.emit()
    
    @pyqtSlot()
    def predictImage(self):
        if not hasattr(self, "net"):
            if not hasattr(self, "fileName"):
                self.text.emit("Load a model and AFM image first!")
                self.finished.emit()
                return
            self.text.emit("Select a network architecture and load a pretrained model first!")
            self.finished.emit()
            return
        elif not hasattr(self, "fileName"):
            self.text.emit("Upload an AFM image first!")
            self.finished.emit()
            return
        
        tf = transforms.Compose([Dataclass.AFMcrop(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        image = Image.open(self.fileName).convert('RGB')
        self.progress.emit(20)
        image = tf(image).float()
        image.unsqueeze_(0)
        image = image.to(self.device)
        self.progress.emit(50)
        
        self.net.eval()
        with torch.no_grad():
            out = self.net(image)
            _, prediction = torch.max(out, 1)
            prediction = "good" if prediction else "bad"
        
        self.text.emit("This AFM image is predicted as a " + prediction + " image!")
        
        self.finished.emit()
 
class Window(QMainWindow):
    loadingModel = pyqtSignal(str, torch.nn.Module)
    uploadingImage = pyqtSignal(str)
    predictingImage = pyqtSignal()
    
    def __init__(self, app):
        super(Window, self).__init__()
        self.dialog = Ui_AFMNet()   
        self.dialog.setupUi(self)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        archs = next(os.walk(self.dirPath + os.path.sep + "models"))[1]
        for arch in archs:
            self.dialog.comboBox.addItem(arch)
        
        self.worker = Worker(self.dirPath, self.device)
        self.dialog.modelButton.clicked.connect(self.loadModel)
        self.dialog.uploadButton.clicked.connect(self.uploadImage)
        self.dialog.predictButton.clicked.connect(self.predictImage)
        app.aboutToQuit.connect(self.closeWindow)
        
    def enableButtons(self, boolean = True):
        self.dialog.modelButton.setEnabled(boolean)
        self.dialog.uploadButton.setEnabled(boolean)
        self.dialog.predictButton.setEnabled(boolean)
        
    def startThread(self, signal, slot):
        if not (hasattr(self, "thread") and isinstance(getattr(self, "thread"), QThread)):
            self.thread = QThread()
        self.worker.progress.connect(self.setProgress)
        self.worker.text.connect(self.setTextLabel)
        self.worker.image.connect(self.setImageLabel)
        self.worker.moveToThread(self.thread)
        signal.connect(slot)
        self.worker.finished.connect(self.finished)
        #self.thread.started.connect(slot)
        self.thread.start()
        
    def setProgress(self, value):
        self.dialog.progressBar.setValue(value)
        
    def setTextLabel(self, string):
        self.dialog.textlabel.setText(string)
        
    def setImageLabel(self, pixmap):
        self.dialog.image.setUnScaledPixmap(pixmap)
 
    def finished(self):
        self.setProgress(100)
        self.enableButtons()
        self.thread.quit()
        
    def loadModel(self):
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Loading...")
        
        self.startThread(self.loadingModel, self.worker.loadModel)
        
        comboText = self.dialog.comboBox.currentText()
        if comboText == "Select Net...":
            self.setTextLabel("Select a network architecture first!")
            self.finished()
            return
        else:
            self.net_arch = comboText.split(' ', 1)[0]
        
        self.modelPath, _ = QFileDialog.getOpenFileName(self, "Load Model", self.dirPath + os.path.sep + "models" + os.path.sep + self.net_arch, "Path Files (*pt *pth)")
        if not self.net_arch.__eq__("ConvNet"):
            arg = eval("models." + self.modelPath.split('/')[-1].split('_')[0])
            self.net = eval("nets." + self.net_arch)(arg)
        else:
            self.net = eval("nets." + self.net_arch)()
            
        self.loadingModel.emit(self.modelPath, self.net)
            
    def uploadImage(self):
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Loading...")
        
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Upload Image", self.dirPath + os.path.sep + "data" + os.path.sep + "val", 'Images (*bmp *gif *jpg *jpeg *png *pbm *tiff *svg *xbm)')
        
        self.startThread(self.uploadingImage, self.worker.uploadImage)
        self.uploadingImage.emit(self.fileName)
        
    def predictImage(self):
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Busy...")
        
        self.startThread(self.predictingImage, self.worker.predictImage)
        self.predictingImage.emit()
        
    def closeWindow(self):
        if hasattr(self, "thread") and isinstance(getattr(self, "thread"), QThread):
            self.thread.quit()
            
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def setUnScaledPixmap(self, pixmap):
        self.unScaledPixmap = pixmap;
        self.setFrameShape(QFrame.NoFrame)
        self.setScaledPixmap()
        
    def setScaledPixmap(self):
        self.scaledPixmap = self.unScaledPixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(self.scaledPixmap)
    
    def resizeEvent(self, event):
        if hasattr(self, "unScaledPixmap"):
            self.setScaledPixmap()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    application = Window(app)
    application.show()
    
    sys.exit(app.exec())