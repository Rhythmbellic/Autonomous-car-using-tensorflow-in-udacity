## Autonomous Car using TensorFlow in Udacity Simulator

This project leverages TensorFlow and OpenCV to develop an autonomous car capable of navigating a simulated environment. The car processes road images captured by the simulator, identifies the optimal path, and autonomously drives within the Udacity simulator. This repository contains the necessary scripts to train the model, process images, and drive the car autonomously.

Project Structure
  behavior.py: Script for training the model using TensorFlow.
  drive.py: Script to drive the car autonomously in the Udacity simulator using the trained model.
  image.py: Script for image processing using OpenCV.
  
Udacity Simulator
The Udacity Self-Driving Car Simulator provides an environment to test and validate the autonomous driving model. The simulator generates road images and driving telemetry which can be used for training and testing the model.

![8c9dd3be-1467-42d4-a1af-b94309e0c72e](https://github.com/Rhythmbellic/Autonomous-car-using-tensorflow-in-udacity/assets/92723976/7140daab-f5a6-4434-a618-5fb91c2e972a)

(I)Collecting the Data
The simplest thing just press record and drive the car
![25d99480-c314-11ea-8e3a-de9b289ab474(1)(1)](https://github.com/Rhythmbellic/Autonomous-car-using-tensorflow-in-udacity/assets/92723976/dae65d76-ff63-49ee-a3f2-2038ca0b7e63)



(II)Training the Model
The behavior.py script contains the code to train a convolutional neural network (CNN) model. The model is trained using images and steering angles recorded by the simulator.

Steps to Train the Model:

1)Load and preprocess data:
  The images and steering angles are loaded from the Udacity simulator's dataset.
  Images are preprocessed using OpenCV functions to crop, resize, and normalize them.

2)Data augmentation:
  Techniques such as zoom, pan, brightness adjustment, and flipping are applied to augment the dataset, improving the model's robustness.

3)Model architecture:
  A CNN architecture inspired by NVIDIA's model for self-driving cars is used.
  The model consists of convolutional layers, dropout layers, and fully connected layers.

4)Training:
  The model is compiled and trained using the mean squared error loss function and the Adam optimizer.
  Training and validation losses are plotted to monitor the training process.

5)Save the model:
  The trained model is saved as model.h5.

(III)Image Processing
The image.py script contains functions for processing images using OpenCV. These functions are used to preprocess images before they are fed into the model for prediction.

Functions:

1)Image Preprocessing:
  Cropping, resizing, and normalizing images.

2)Data Augmentation:
  Functions to apply zoom, pan, brightness adjustment, and flipping to images.


(IV)Autonomous Driving
The drive.py script connects to the Udacity simulator and uses the trained model to drive the car autonomously.

Steps:

1)Setup server:
  A Flask server is set up with SocketIO to communicate with the simulator.

2)Load the model:
  The trained model (model.h5) is loaded.

3)Process images and predict steering angles:
  Images from the simulator are processed using the same preprocessing steps as during training.
  The model predicts the steering angle for each processed image.

4)Control the car:
  The predicted steering angle and a throttle value are sent back to the simulator to control the car.  

![147305077-8b86ec92-ed26-43ca-860c-5812fea9b1d8](https://github.com/Rhythmbellic/Autonomous-car-using-tensorflow-in-udacity/assets/92723976/306488e2-6626-4f3d-9c7b-0395bf945215)
