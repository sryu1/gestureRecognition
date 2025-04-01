#Import the necessary packages
import cv2
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands()


def main():
    #Initialize video capture in a variable
    feed = cv2.VideoCapture(0)

    #Double check that we can open the camera
    if (feed.isOpened()):
        print("The camera is successfully opened")
    else:
        print("Could not open the camera") 

    #Enter the infinite while loop here
    while True:
        #Read in a frame of the webcam feed and store it
        success, img = feed.read()
        if not success:
            print("Not able to read the frame.")
            break
        
        
        
        #Flip camera so that the display acts like a mirror
        img = cv2.flip(img, 1)
        
        #Convert the color space from RGB to OpenCV recognised BGR
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Recognise the hands from the frame
        recHands = hands.process(img)
        if recHands.multi_hand_landmarks:
            for hand in recHands.multi_hand_landmarks:
                #Draw dots to visualise the landmarks
                for datapoint_id, point in enumerate(hand.landmark):
                    h, w, c = img.shape
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(img, (x, y),
                               5, (250, 200, 50)
                               , cv2.FILLED)

        #Display frames to the output feed
        cv2.imshow('Camera Feed',img)
        
        #Logic to handle program termination by user inp
        if cv2.waitKey(1) == ord('q'):
            feed.release()
            cv2.destroyAllWindows()
            print("Terminating Program")
            exit()

    feed.release()
    cv2.destroyAllWindows()

main()
