# self-driving-car
Self-Driving Car Model using CNN and Data Augmentation
This project implements a self-driving car model using a convolutional neural network (CNN) trained on simulated driving data. The model predicts steering angles based on input images from a front-facing camera, allowing the vehicle to autonomously navigate a track.
Data Preprocessing: Cleans and balances the dataset to prevent steering angle bias.

Data Augmentation: Includes random zoom, panning, brightness adjustments, and horizontal flipping to improve generalization.
CNN Architecture: A deep learning model inspired by NVIDIA's self-driving car architecture.
Training & Validation: Uses Mean Squared Error (MSE) loss and the Adam optimizer to minimize steering prediction errors.
Model Saving: Saves the trained model for later use in real-time autonomous driving applications.

# Dataset
The dataset consists of images and a driving_log.csv file containing steering angles and associated image paths. The images are captured from three cameras: center, left, and right.
Data Preprocessing
Path Cleaning: Removes directory prefixes from image paths.
Histogram Equalization: Balances the dataset by limiting the number of samples per bin.

# Image Processing:
Crops unnecessary parts of the image.
Converts RGB to YUV (preferred for self-driving models).
Applies Gaussian blur for noise reduction.
Resizes images for efficient model training.
Normalizes pixel values.

# The model follows a CNN-based architecture:
Five convolutional layers with ELU activation.
Dropout layers to prevent overfitting.
Fully connected layers for steering angle regression.
Adam optimizer with learning_rate=1e-3.

# Author 
Vishvender Tyagi
