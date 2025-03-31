# Documentation

> [!IMPORTANT]
> Some information below are incorrect. This documentation will need to be updated.

## Introduction to the Project

Welcome to the CodeNetwork 2025 Semester 1 gesture recognition program. Here is all the necessary documentation you will need to operate and understand the gesture recognition program.

This project focuses on detecting distinct live hand gestures. A Machine Learning approach is used to distinguish gestures.

## Data Collection

### Step 1

- Get a camera (any camera)

### Step 2

- Start taking photos of your gesture
- Make sure to grab different angles (move the camera around and move a hand around)
- Try to grab different lightings (time of day, different directional lighting)
- Try different backgrounds
- Try different distances from the camera
- If you can try to capture other people, different skin tones, small hands, big hands, etc.

### Step 3

- Upload images to the [Hugging face repository](https://huggingface.co/datasets/CNGR/CN_Gesture_Recognition/tree/main)
- Depending on the gestures you captured add to the respective folder

## Data Annotation
### Step 1

- Clone the HuggingFace repository using the following command:
``` bash
git clone https://huggingface.co/datasets/CNGR/CN_Gesture_Recognition --depth=1
```
- This will download all the pictures from our dataset, organised into their respective folders.
- It would be wise to just select one gesture (folder) to work on at a time 

### Step 2

- Go to cvat.ai and log in with your GitHub account (or another way, but GitHub will be convenient since we've already been using it a lot).
- On the landing page, select 'create new task' option.
- Name the task after the gesture you picked in step 1
- Select 'add label' and give it an appropriate name
- Now follow the prompt to upload the files and upload the photos associated with your gesture (enter the folder for you gesture and hit CTRL + A to select them all, then upload)
- Now just select 'submit and open'

### Step 3

- In the navbar up the top, select 'jobs' then select your job
- You should see one of the pictures of your gesture. On the sidebar to the left, select the rectangle icon and make sure the label is set to the one you created.
- Draw a new rectangle that highlights the section of the image that your gesture is in. (Note that you can press 'N' to start and stop the rectangle drawing process, and 'F' to progress to the next image. Utilising this will drastically improve the speed at which you annotate

## File Structure

For the YOLO model, the following file structure is required:

For CNN:

<br><br><br><br><br><br><br><br><br><br>

## Platform

## Libraries

## Hardware

This project requires a webcam.

## Running the Program

Run the gesture-recog.py file in order to launch the main program.

## Main Loop

- Continuously capture the camera input. *Flip the input on the vertical axis so the output reflects like a mirror
- Read the frames into variables
- Loop ends upon user input (video capture stops asw)

## Hand Recognition

- Using MediaPipe and OpenCV
  
### Using Contour Analysis

- Objects to store and update hand data
- Convert to greyscale and smoothen to facilitate optimal edge detection (ignore first 30 frames as cameras generally use this time to adjust to shadows and lighting)
- Threshold the hand from the background
- Use contour analysis to extract the contours, find the largest contour and compute the convex hull
  
## Gesture Recognition

### Neural Network

- Use the keras convolutional networks
