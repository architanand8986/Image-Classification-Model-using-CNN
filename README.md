# CIFAR-10 Image Classification with PyTorch

This repository contains the implementation of an image classification model for the CIFAR-10 dataset using PyTorch. The model is built using Convolutional Neural Networks (CNNs) and utilizes data augmentation, learning rate scheduling, and early stopping to achieve improved performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Training Setup](#training-setup)
- [Installation](#installation)
- [Results](#results)


## Overview

The goal of this project is to classify images from the CIFAR-10 dataset into 10 different categories using a deep learning model. The model is built with PyTorch and incorporates techniques such as TrivialAugmentWide for data augmentation, Adam optimizer, ReduceLROnPlateau for dynamic learning rate adjustment, and early stopping to prevent overfitting.

## Dataset

The **CIFAR-10** dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. You can find more information about the dataset [here](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).


### Classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck


## Training Setup

- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam (learning rate = 0.001)
- **Scheduler**: ReduceLROnPlateau (adjusts learning rate based on validation loss)
- **Early Stopping**: Stops training if validation performance does not improve for a number of epochs

### Hyperparameters:

- Number of epochs: 20
- Learning rate: 0.001

## Results

The model achieved an accuracy of **83%** on the test dataset.

| Metric            | Value        |
|-------------------|--------------|
| Training Accuracy  | 78%          |
| Test Accuracy      | 83%          |

![Confusion Matrix](https://github.com/architanand8986/Image-Classification-Model-using-CNN/blob/main/Image/Confusion%20Matrix.png)





