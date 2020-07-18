# -*- coding: utf-8 -*-
"""
Defines all functionalities of the GUI, such as the response of clicking on the buttons and parrallel processing.
"""

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel, QFrame
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
import torch
from torchvision import transforms
from PIL import Image
import os
import pickle

## To generate docs using Sphinx + autodoc; comment these out
from dialog import Ui_AFMNet  # importing our generated file
import nets
from Dataclass import Dataclass

class Worker(QObject):
    """
    A worker class which can be called in a separate thread.
    As a result the worker can do things, while the GUI runs on another thread. As a result the GUI doesn't freeze.
    """
    
    ## Initialize different signals which can be can be sent to the Window class
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    text = pyqtSignal(str)
    image = pyqtSignal(QPixmap)
    
    def __init__(self, dirPath, device):
        """
        Initializes a (parallel) worker

        Args:
            dirPath (str): Path to the project directory.
            device (device): The device which is used by pytorch.

        Returns:
            None.

        """
        super().__init__()
        
        ## Safe directory and device to the class
        self.dirPath = dirPath
        self.device = device
    
    ## Define method as a slot
    @pyqtSlot(str, torch.nn.Module)
    def loadModel(self, modelPath, net):
        """
        Let's the worker load the model from the selected path file.
        
        Args:
            modelPath (str): The path to the selected file.
            net (Model): The network object created in the window class.

        Returns:
            None.

        """
        
        ## Update progress bar
        self.net = net
        self.progress.emit(25)
        
        ## Load the network parameters to the network object and move the net to the desired device
        try:
            self.mean, self.std = nets.load(net, modelPath, device = self.device)
            self.net.to(self.device)
        except RuntimeError:
            ## This exception is catched when a model with the wrong architecture is selected
            ## i.e. ConvNet is selected in the dropdown menu, but a FineNet .pt file is opened
            self.text.emit("This model does not have a " + type(self.net).__name__ + " architecture! Choose again!")
            self.finished.emit()
            return
        except (FileNotFoundError, pickle.UnpicklingError):
            ## This exception is catched when a wrong file type is opened
            self.text.emit("File not supported!")
            self.finished.emit()
            return
        
        ## Set progress bar to 70%
        self.progress.emit(70)
        
        ## Check if an image is already loaded in the GUI and update the text label accordingly
        if hasattr(self, "fileName"):
            self.text.emit("You can now use the model to predict whether the AFM image is good or bad")
        else:
            self.text.emit("Upload an AFM image!")
        
        ## Sent the finished signal to close the thread
        self.finished.emit()
    
    ## Define method as a slot
    @pyqtSlot(str)
    def uploadImage(self, fileName):
        """
        Let's the worker load the AFM image from the selected path.

        Args:
            fileName (str): The path to the selected file.

        Returns:
            None.

        """
        
        ## Save filename (path) in worker class
        self.fileName = fileName
        
        ## Set progress to 30%
        self.progress.emit(30)
        
        ## Aborting when trying to upload an unsupported file
        if not (self.fileName.endswith('.bmp') or self.fileName.endswith('.gif') or self.fileName.endswith('.jpg') 
                or self.fileName.endswith('.jpeg') or self.fileName.endswith('.png') or self.fileName.endswith('.pbm')
                or self.fileName.endswith('.tiff') or self.fileName.endswith('.svg') or self.fileName.endswith('.xbm')):
            self.text.emit("File not supported!")
            self.finished.emit()
            return
        
        ## Set progress to 50%
        self.progress.emit(50)
        
        ## Create a pixmap from the image
        pixmap = QPixmap(self.fileName)
        
        ## Set progress to 80%
        self.progress.emit(80)
        
        ## Sent the image signal to the Windows class
        ## setImageLabel method from the Windows class is called
        self.image.emit(pixmap)
        
        ## Check if a model is already loaded in the GUI and update the text label accordingly
        if not hasattr(self, "net"):
            self.text.emit("Select a network architecture and load a pretrained model.")
        else:
            self.text.emit("You can now use the model to predict whether the AFM image is good or bad!")
        
        ## Sent the finished signal to close the thread
        self.progress.emit(100)
        self.finished.emit()
    
    ## Define method as a slot
    @pyqtSlot()
    def predictImage(self):
        """
        Let's the worker predict if the loaded AFM image is 'good' or 'bad'.

        Returns:
            None.

        """
        
        ## Check if a model and image is loaded
        ## If not, prompt the user to load a pretrained model and/or AFM image
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
        
        ## The transforms used to preprocess the image
        tf = transforms.Compose([Dataclass.AFMcrop(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        
        ## Open the image as a PIL image
        image = Image.open(self.fileName).convert('RGB')
        
        ## Set the progress bar to 20%
        self.progress.emit(20)
        
        ## Preprocess the image and move to the desired device
        image = tf(image).float()
        image.unsqueeze_(0)
        image = image.to(self.device)
        
        ## Set the progress bar to 50%
        self.progress.emit(50)
        
        ## Set the network in evaluation mode
        self.net.eval()
        
        ## Perform a forward pass and retrieve the prediction/classification
        with torch.no_grad():
            out = self.net(image)
            _, prediction = torch.max(out, 1)
            prediction = "good" if prediction else "bad"
        
        ## Upload the text label corresponding to the prediction
        self.text.emit("This AFM image is predicted as a " + prediction + " image!")
        
        ## Sent the finished signal to close the thread
        self.finished.emit()
 
class Window(QMainWindow):
    """
    A class that connect functionality to the GUI window. Starts a new thread with a worker when clicking on a button.
    """
    
    ## Initialize different signals which can be sent to the Worker class
    loadingModel = pyqtSignal(str, torch.nn.Module)
    uploadingImage = pyqtSignal(str)
    predictingImage = pyqtSignal()
    
    def __init__(self, app):
        """
        Initializes the window, initializes a (parallel) worker and connects all the functionalities to the dialog.

        Args:
            app (QApplication): A QApplication object of the GUI.

        Returns:
            None.

        """
        
        ## Create a window
        super(Window, self).__init__()
        
        ## Load lay-out of GUI from the generated dialog file
        self.dialog = Ui_AFMNet()   
        self.dialog.setupUi(self)
        
        ## Get the directory of the AFMNet folder
        self.dirPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".."
        
        ## Set the device for Pytorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ## Add the model architectures (ConvNet, FineNet and FeatureNet) to the dropdown menu
        archs = next(os.walk(self.dirPath + os.path.sep + "models"))[1]
        for arch in archs:
            self.dialog.comboBox.addItem(arch)
        
        ## Initialize a (parallel) worker and pass the directory and device
        self.worker = Worker(self.dirPath, self.device)
        
        ## Connect functions to the different buttons
        self.dialog.modelButton.clicked.connect(self.loadModel)
        self.dialog.uploadButton.clicked.connect(self.uploadImage)
        self.dialog.predictButton.clicked.connect(self.predictImage)
        
        ## Close all threads when the window closes
        app.aboutToQuit.connect(self.closeWindow)
        
    def enableButtons(self, boolean = True):
        """
        Enables or disabled all buttons.

        Args:
            boolean (bool, optional): All buttons will be enabled if 'True', they will be disabled if 'False'. Defaults to True.

        Returns:
            None.

        """
        self.dialog.modelButton.setEnabled(boolean)
        self.dialog.uploadButton.setEnabled(boolean)
        self.dialog.predictButton.setEnabled(boolean)
        
    def startThread(self, signal, slot):
        """
        Creates and starts a thread for the (parallel) worker. Also connects a signal to a slot of the worker.

        Args:
            signal (pyqtSignal): A signal attribute of the Window class.
            slot (pyqtSignal): A slot attribute of the Worker class.

        Returns:
            None.

        """
        ## Start a QThread if it does not exist yet
        if not (hasattr(self, "thread") and isinstance(getattr(self, "thread"), QThread)):
            self.thread = QThread()
            
        ## Connect the signals of the worker to methods in the window class
        self.worker.progress.connect(self.setProgress)
        self.worker.text.connect(self.setTextLabel)
        self.worker.image.connect(self.setImageLabel)
        
        ## Move the worker to the thread
        self.worker.moveToThread(self.thread)
        
        ## Connect a sginal of the worker to a method in the window class
        signal.connect(slot)
        self.worker.finished.connect(self.finished)
        
        ## Start the thread
        self.thread.start()
        
    def setProgress(self, value):
        """
        Sets the progress of the progress bar to a certain value.

        Args:
            value (int): The progress bar will be set to this (percentage) value.

        Returns:
            None.

        """
        self.dialog.progressBar.setValue(value)
        
    def setTextLabel(self, string):
        """
        Sets the text label of the text beneath the image.

        Args:
            string (str): String of desired text for the label.

        Returns:
            None.

        """
        self.dialog.textlabel.setText(string)
        
    def setImageLabel(self, pixmap):
        """
        Updates the image label with the provided pixmap.

        Args:
            pixmap (QPixmap): A pixmap of an AFM image.

        Returns:
            None.

        """
        self.dialog.image.setUnScaledPixmap(pixmap)
 
    def finished(self):
        """
        Called when a thread is finished. Sets the progress to 100, enables all buttons and quits the thread.

        Returns:
            None.

        """
        self.setProgress(100)
        self.enableButtons()
        self.thread.quit()
        
    def loadModel(self):
        """
        Method connected to the 'Load Model' button. Loads a model architecture.

        Returns:
            None.

        """
        
        ## Set the progress to 0, disable all buttons and set the text label to "Loading..."
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Loading...")
        
        ## Start a thread which connects the loadingModel signal to the LoadModel slot of the worker
        self.startThread(self.loadingModel, self.worker.loadModel)
        
        ## Retrieve the selected network architecture
        comboText = self.dialog.comboBox.currentText()
        if comboText == "Select Net...":
            ## If no network architecture is selected, prompt the user to select a network architecture.
            self.setTextLabel("Select a network architecture first!")
            self.finished()
            return
        else:
            self.net_arch = comboText.split(' ', 1)[0]
        
        ## Open an explorer window in which you can select a model of architecture net_arch
        self.modelPath, _ = QFileDialog.getOpenFileName(self, "Load Model", self.dirPath + os.path.sep + "models" + os.path.sep + self.net_arch, "Path Files (*pt *pth)")
        
        ## Initialize the network (with a class from nets.py)
        if not self.net_arch.__eq__("ConvNet"):
            arg = eval("models." + self.modelPath.split('/')[-1].split('_')[0])
            self.net = eval("nets." + self.net_arch)(arg)
        else:
            self.net = eval("nets." + self.net_arch)()
        
        ## Send a signal with the path and network to the loadModel slot of the worker
        self.loadingModel.emit(self.modelPath, self.net)
            
    def uploadImage(self):
        """
        Method connected to the 'Upload Image' button. Opens an AFM image to predict.

        Returns:
            None.

        """
        
        ## Set the progress to 0, disable all buttons and set the text label to "Loading..."
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Loading...")
        
        ## Open an explorer window in which you can open an AFM image
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Upload Image", self.dirPath + os.path.sep + "data" + os.path.sep + "val", 'Images (*bmp *gif *jpg *jpeg *png *pbm *tiff *svg *xbm)')
        
        ## Start a thread which connects the uploadingImage signal to the uploadImage slot of the worker
        self.startThread(self.uploadingImage, self.worker.uploadImage)
        
        ## Send a signal with the path to the uploadImage slot of the worker
        self.uploadingImage.emit(self.fileName)
        
    def predictImage(self):
        """
        Method connected to the 'Predict Image' button. Classifies an image as 'Good' or 'Bad'.

        Returns:
            None.

        """
        ## Set the progress to 0, disable all buttons and set the text label to "Busy..."
        self.setProgress(0)
        self.enableButtons(False)
        self.setTextLabel("Busy...")
        
        ## Start a thread which connects the predictingImage signal to the predictImage slot of the worker
        self.startThread(self.predictingImage, self.worker.predictImage)
        
        ## Send a signal to the predictImage slot of the worker
        self.predictingImage.emit()
        
    def closeWindow(self):
        """
        Closes the remaining threads.

        Returns:
            None.

        """
        if hasattr(self, "thread") and isinstance(getattr(self, "thread"), QThread):
            self.thread.quit()
            
class ImageLabel(QLabel):
    """
    A custom label used for images. The image label instance is created in dialog.py.
    """
    
    def __init__(self, parent=None):
        """
        Initializes an image label.

        Args:
            parent (QWidget, optional): The widget in which the label is placed. Defaults to None.

        Returns:
            None.

        """
        super().__init__(parent)
    
    def setUnScaledPixmap(self, pixmap):
        """
        Sets an unscaled Pixmap.

        Args:
            pixmap (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        
        ## Set the unscaled pixamp and frame shape
        self.unScaledPixmap = pixmap;
        self.setFrameShape(QFrame.NoFrame)
        
        ## Rescale the image to fit in the image label
        self.setScaledPixmap()
        
    def setScaledPixmap(self):
        """
        Scales the unscaled pixmap to fit in the frame of the image label.

        Returns:
            None.

        """
        
        ## Scale the image to fit in the label
        self.scaledPixmap = self.unScaledPixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        
        ## Set the scaled pixmap as the pixmap (inherited method from QLabel)
        self.setPixmap(self.scaledPixmap)
    
    def resizeEvent(self, event):
        """
        Method that resizes the image label when resizing the window.

        Args:
            event (QEvent): An event, such as rescaling the window.

        Returns:
            None.

        """
        if hasattr(self, "unScaledPixmap"):
            self.setScaledPixmap()