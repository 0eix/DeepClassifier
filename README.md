# Flower Image Classifier: An AI Programming Project

This project is part of the Udacity AI Programming with Python Nanodegree, where a deep learning model is developed for identifying 102 different species of flowers based on their images. The project is divided into two parts: 

1. Designing and training the model using Jupyter notebook.
2. Turning the trained model into a command-line application that can train on any set of images, and make predictions on new images.

The final command-line application allows customization of various aspects including CNN architecture selection, setting hyperparameters, choosing between GPU/CPU, and saving the training to a checkpoint to resume later.

## File Structure:

Here is an overview of the purpose and usage of each file in this project.

1. **cli_utils.py**: Contains helper functions for parsing command-line arguments. This aids in customizing the application's functionality like selecting the CNN architecture, setting hyperparameters, deciding whether to use GPU, and more.

2. **data_utils.py**: Responsible for data preprocessing and loading tasks. It includes functions to load image datasets from a directory, split them into training, validation and testing sets, and apply necessary transformations to prepare data for the model.

3. **device_utils.py**: Includes utility functions for device selection. It helps to identify whether a GPU is available and should be used for training and inference.

4. **model_utils.py**: Contains functions to build and load the deep learning model. This includes creating a classifier using one of the pre-trained models (AlexNet, VGG, DenseNet, etc.), and loading a checkpoint to resume training.

5. **predict.py**: This script uses the trained model to make predictions on new images. It processes the input image, performs inference using the model, and outputs the top K most likely classes along with their probabilities.

6. **train.py**: Contains the functionality required for training the model. It defines the training loop, validation loop, and handles saving the model's state to a checkpoint after each epoch.

## How to Run:

You can train a new network on a data-set with `train.py` and predict the class for an input image with `predict.py`. Run `python train.py -h` and `python predict.py -h` to see the available command-line options for each script.

## Requirements:

The application is built using Python and requires PyTorch, NumPy, and Pillow libraries.

Remember to install these dependencies before running the application.
