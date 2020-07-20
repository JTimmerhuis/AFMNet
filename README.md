# AFMNet
Convolutional Classifier which can classify AFM images as either 'good' or 'bad'.

This project consists of code to train a convolutional neural network as a classifier of AFM images with relatively little training data. A GUI application in which you can load a pretrained model and use it on an AFM image has been used as well. We used PyTorch to train the NN and PyQT5 to make the GUI.

## PyTorch
[PyTorch] is an open source machine learning framework for Python. Their [tutorial] page is a great place to start with PyTorch. We used parts of the [Training a Classifier] and [Transfer Learning for Computer Vision] tutorials.

## File structure
The file structure of the repository is shown in the tree below. We will discuss every folder and their contents below.

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

## `data/`
All the data is stored in the `data` folder.



[PyTorch]: https://pytorch.org/
[tutorial]: https://pytorch.org/tutorials
[Training a Classifier]: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
[Transfer Learning for Computer Vision]: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
[PyTorch DataLoader]: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
