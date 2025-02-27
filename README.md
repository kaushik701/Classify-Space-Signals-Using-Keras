# SETI Space Signal Classification Project
This project implements a Convolutional Neural Network (CNN) to classify SETI (Search for Extraterrestrial Intelligence) signals into 4 categories: squiggle, narrowband, noise, and narrowbanddrd.

# Dataset Structure
The dataset is organized as follows:

Training set: 3,200 images
Validation set: 800 images
Image dimensions: 64x128 pixels (grayscale)
The image files are compressed for ease to upload
4 signal classes
![image](https://github.com/user-attachments/assets/221b8992-678f-4583-9cf4-0ef22737e8ae)


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
![image](https://github.com/user-attachments/assets/9f8a3c80-afe5-4054-9064-9dd1e5562f5a)


Classification metrics reporting

Data augmentation for better generalization

# Usage
Prepare your dataset in CSV format (images.csv and labels.csv)

Run the training script

Model weights will be saved as "model_weights.h5"

Evaluate model performance using confusion matrix and classification report

# Performance Visualization
The code includes:
![image](https://github.com/user-attachments/assets/a50eee14-d812-495d-acd3-a591b017906b)


Training/validation loss curves

Confusion matrix heatmap

Classification metrics including accuracy, precision, recall, and F1-score
