# Documentation

## Introduction to the Project
Welcome to the CodeNetwork 2025 Semester 1 gesture recognition program. Here is all the necessary documentation you will need to operate and understand the gesture recognition program.


This project focuses on detecting distinct live hand gestures. A Machine Learning approach is used to distinguish gestures.

## Platform

## Libraries

## Hardware
This project requires a webcam.

## Running the Program
Run the gesture-recog.py file in order to launch the main program.

## Data Collection
* Capture different gestures in different environments

## Main Loop
* Continuously capture the camera input. *Flip the input on the vertical axis so the output reflects like a mirror
* Read the frames into variables
* Loop ends upon user input (video capture stops asw)

## Hand Recognition
* Using MediaPipe and OpenCV
  
### Using Contour Analysis
* Objects to store and update hand data
* Convert to greyscale and smoothen to facilitate optimal edge detection (ignore first 30 frames as cameras generally use this time to adjust to shadows and lighting)
* Threshold the hand from the background
* Use contour analysis to extract the contours, find the largest contour and compute the convex hull
  
## Gesture Recognition
### Neural Network
* Use the keras convolutional networks 

