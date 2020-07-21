# AFMNet
This project consists of code to train a convolutional neural network as a classifier of AFM images with relatively little training data. The AFM images are classified as either 'good' or 'bad'. A GUI application in which you can load a pretrained model and use it on an AFM image has been used as well. We used PyTorch to train the NN and PyQT5 to make the GUI.

## PyTorch
[PyTorch] is an open source machine learning framework for Python. Their [tutorial] page is a great place to start with PyTorch. We used parts of the [Training a Classifier] and [Transfer Learning for Computer Vision] tutorials.

## Git LFS
Images (`*.png` and `*.jpg`) and model (`*.pt`) files are pushed to Git using [Git Large File Storage] (Git LFS). Git LFS replaces large files with text pointers inside Git, whereas the contents are pushed to a remote account. My account has only '1 GB' LFS storage, so only the best models are uploaded to git. There exists a `*.zip` file somewhere with all the trained models.

## File structure
The file structure of the repository is shown in the tree below. We will discuss every folder and their contents below. We will first discuss the folders unrelated to the GUI and discuss the GUI separately in the end.

    .
    ├── code
    │   ├── Dataclass.py
    │   ├── functions.py
    │   └── nets.py
    ├── data
    │   ├── train
    │   │   ├── bad
    │   │   │   ├── badTestImage1
    │   │   │   ├── badTestImage2
    │   │   │   └── ...
    │   │   └── good
    │   │   │   ├── goodTestImage1
    │   │   │   ├── goodTestImage2
    │   │   │   └── ...
    │   └── val
    │   │   ├── bad
    │   │   │   ├── badValidationImage1
    │   │   │   ├── badValidationImage2
    │   │   │   └── ...
    │   │   └── good
    │   │   │   ├── goodValidationImage1
    │   │   │   ├── goodValidationImage2
    │   │   │   └── ...
    ├── gui
    │   ├── application.py
    │   ├── dialog.py
    │   └── dialog.ui
    ├── models
    │   ├── ConvNet
    │   │   ├── ConvNetModel1
    │   │   ├── ConvNetModel2
    │   │   └── ...
    │   ├── FeatureNet
    │   │   ├── FeatureNetModel1
    │   │   ├── FeatureNetModel2
    │   │   └── ...
    │   ├── FineNet
    │   │   ├── FineNetModel1
    │   │   ├── FineNetModel2
    │   │   └── ...
    │   └── info.txt
    ├── AFMNet.py
    └── app.py
    
## `code/` and `AFMNet.py`
Everything relevant to training the Neural Networks is in the `code` folder and `AFMnet.py`. We will discuss each file below.

### `AFMNet.py`
`AFMNet.py` is the most important file to run; it imports all important code from the `code` folder. It can be used to train a new model or load a pretrained model. Furthermore, you can visualise the performance of the model. Finally, you can save a model. Be careful with not overwriting a pre-existing model, though!! Commented lines are alternatives to uncommented line, such as using a different optimizer.

### `code/Dataclass.py`
`Dataclass.py` has a single class `Dataclass()`. This class has methods that manipulate the data used to either train or validate the models. It stores the transforms needed for preprocessing the data and can load the data using the [PyTorch DataLoader].

### `code/functions.py`
`functions.py` has methods to train a model and to visualise the performance of a model.

### `code/nets.py`
`nets.py` has classes for the three module types used in this project: ConvNet, FeatureNet and FineNet. It also had methods which can be used to save the parameters of the model or load a pretrained model.

### `data/ folder`
All the data is stored in the `data` folder. The data is divided in a training data and validation data. There are 60 images (32 good, 28 bad) in the training data set and 21 (12 good, 9 bad) in the validation data set. These have been manually selected. As said before, the [PyTorch DataLoader] is used to load the data. This means that the folders inside the `train` or `val` have to be named according to the labels, i.e. `bad` and `good`, with their corresponding images inside. It is important the the data is placed in this folder structure, otherwise the code won't work.

### `models/ folder`
The models folders hostst all the models. Each folder contains models corresponding to their model type. `info.txt` contains information about the convolutional netowork we designed ourselves, `ConvNet`. Each model is saved as a `*.pt` file and has a very specific naming convention:

* If the model is a `FeatureNet` of `FineNet`, the file name will start with the architecture of the pre-trained model. Currently, only `resnet18` is supported.
* Next, the optimizer, or training algorithm, will be in the filename. The exact naming convention depends on the batch size.
    * For a batch size of 1, it will be the name of the optimizer. For instance, `Adam` for the Adam algorithm and `SGD` for stochastic gradient descent.
    * For a batch size greater than 1, it will be `Batch` + the name of the optimizer + the batch size. For instance, `BatchAdam4`. An exception is made for Stochastic Gradient Descent (because Stochastic implies a batch size of 1): is is then named `BGD` + batch size, e.g. `BGD4`.
* Next comes the name of the used PyTorch scheduler, e.g. `StepLR` or `ReduceLROnPlateau`. `NoScheduler` is used in case no scheduler was used during training.
* Then the normalization mean and standard deviation used to normalize the images can be deduced in the filename. `ImageMStd` is used when the images were normalized using the mean and standard deviation of the [ImageNet] dataset during training. This is always the case for `FeatureNet` and `FineNet`. `OwnMStd` is used when the images were normalized by the mean and standard deviation of out own dataset.
* Next you will see `Eps` plus the Epoch number in the file name. For instance, `Eps50`.
* Finally, you will see the validation accuracy as `Val` + the accuracy in percentages rounded to the nearest integer, e.g. `Val86`

Between each bullet point an underscore will be present in the filename. An example for a Convnet would be `Adam_NoScheduler_OwnMStd_Eps25_Val71.pt`. An example for a FineNet or FeatureNet would be `resnet18_Adam_NoScheduler_ImageMStd_Eps50_Val81.pt`. This naming convention is used by the methods in `code/nets.py`.

## GUI
Below we will describe how to run the GUI, how to use the GUI and we will finally discuss the code behind the GUI. The GUI is a great way to visualise the prediction/classification of the models.

### Running the GUI
The code that runs the GUI is `app.py` in the main directory. It is recommended to run this from the (Anaconda) command line in the working directory

```
python app.py
```

This opens the application. The code can also be run from an IDE, such as Spyder, however because IDEs are GUIs themselves, this might give weird errors.

### Using the GUI
Below you see a screenshot of the GUI.

![gui]

The GUI has three buttons: one to load a pre-trained model, one to load an AFM image and one to predict the label of that image. To load a pre-trained model one must first select a model type -- `ConvNet`, `FeatureNet`, or `FineNet` -- in the `Select Net...` dropdown menu. You can then click on the `Load a:` button to load a pre-trained model. One can use the `Upload Image` button to upload an image. Finally, you can use the `Predict Image` button to predict if the image is a 'good' or 'bad' AFM. This button will not work if either the model or image is not loaded.

### Behind the GUI
The GUI is created using the [PyQt5] module. You can use their designer to design a GUI and then use python to add functionalities to e.g. the buttons.

#### QT Designer and `gui/dialog.py`
After installing PyQt5, you can open the designer by simply typing `designer` in the (Anaconda) command prompt. The design of the GUI is saved as a `.ui` file, in our case `gui/dialog.ui`. The lay-out can be changed in the Qt Designer. It is then possible to convert the `*.ui` file to a python file by typing the following command in the command prompt (while in the gui folder):

```
pyuic5 dialog.ui -o dialog.py
```

This generates a `dialog.py` file. In our case, to make the GUI work, we have to change line 139 of `dialog.py` from

```
from application import ImageLabel
```
to
```
from gui.application import ImageLabel
```

#### `gui/application.py`
`application.py` is the file that adds functionality to the labels present in the desginer and `dialog.py`. It uses a `Window()` class to add these functionalities. Often a `Worker()` class is started on a parallel thread separate from the thread the GUI is running on. The `Worker()` then carries out the button functionalities. If you don't do this, the whole GUI would freeze until one functionality, for instance loading an image, is finished.

[Git Large File Storage]: https://git-lfs.github.com/
[PyTorch]: https://pytorch.org/
[tutorial]: https://pytorch.org/tutorials
[Training a Classifier]: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
[Transfer Learning for Computer Vision]: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
[PyTorch DataLoader]: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
[ImageNet]: http://www.image-net.org/
[PyQt5]: https://www.riverbankcomputing.com/static/Docs/PyQt5/
[gui]: gui/gui.png
