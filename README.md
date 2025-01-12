# CNN and Transfer Learning with EfficientNet

## 📋 Project Description

This project demonstrates image classification using a custom Convolutional Neural Network (CNN) and EfficientNet with transfer learning. The primary goal is to classify animal images into 10 categories from the Animals-10 dataset. The project involves two parts:

* Custom CNN Model: A CNN built from scratch with six convolutional layers and dropout regularization.

* Transfer Learning with EfficientNet: Fine-tuning a pre-trained EfficientNet model to adapt to the dataset.

## 🚀 Features

* Custom CNN architecture with detailed training and validation steps.

* Implementation of dropout layers to improve generalization.

* Transfer learning with EfficientNet using two approaches:

* Training only the fully connected (FC) layer.

* Training the last two convolutional blocks and the FC layer.

* Visualization of training metrics (loss and accuracy).

* Evaluation using confusion matrices and comparison of results.

## 🛠️ Technologies Used

* Python: Programming language

* PyTorch: Deep learning framework for building and training models

* NumPy: Numerical computations

* Matplotlib: Visualization of training metrics

* scikit-learn: Evaluation metrics and utilities

## 🔍 Technical Details

### Custom CNN

* Layers: 6 convolutional layers with ReLU activation and MaxPooling.

* Regularization: Dropout added to improve generalization.

* Optimizer: Adam with a learning rate of 0.001.

* Loss Function: CrossEntropyLoss for multi-class classification.

### Transfer Learning with EfficientNet

* Pre-trained Model: EfficientNet-B0 initialized with ImageNet weights.

* Fine-Tuning:

* Case 1: Trained only the FC layer.

* Case 2: Trained the last two convolutional blocks and the FC layer.

* Optimizer: Adam with a learning rate of 0.001.

### Evaluation

* Validation and test accuracy measured.

* Confusion matrices generated to visualize model performance.
