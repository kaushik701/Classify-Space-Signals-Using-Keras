# SETI Space Signal Classification Project
This project implements a Convolutional Neural Network (CNN) to classify SETI (Search for Extraterrestrial Intelligence) signals into 4 categories: squiggle, narrowband, noise, and narrowbanddrd.

# Dataset Structure
The dataset is organized as follows:

Training set: 3,200 images
Validation set: 800 images
Image dimensions: 64x128 pixels (grayscale)
4 signal classes

# Requirements
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
livelossplot

# Model Architecture
The CNN model consists of:

Multiple Convolutional layers with 32 and 64 filters

Batch Normalization layers

ReLU activation

MaxPooling layers

Dropout layers for regularization

Dense layers with final softmax activation for 4-class classification

# Training Details
Optimizer: Adam with exponential learning rate decay
Initial learning rate: 0.005
Batch size: 32
Epochs: 12
Data augmentation: Horizontal flipping
Loss function: Categorical crossentropy

# Features
Real-time loss plotting during training

Model checkpoint saving

Confusion matrix visualization

Classification metrics reporting

Data augmentation for better generalization

# Usage
Prepare your dataset in CSV format (images.csv and labels.csv)

Run the training script

Model weights will be saved as "model_weights.h5"

Evaluate model performance using confusion matrix and classification report

# Performance Visualization
The code includes:

Training/validation loss curves

Confusion matrix heatmap

Classification metrics including accuracy, precision, recall, and F1-score
