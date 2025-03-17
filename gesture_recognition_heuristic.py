#Import the necessary packages

#Initialize video capture in a variable

#Enter the infinite while loop here

    #Read in a frame of the webcam feed and store it

    #logic to handle 

import cv2
import mediapipe as mp
import math
import numpy as np
import platform
import sys
import time

class HandGestureRecognizer:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tip IDs
        self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        
        # Gesture database
        self.gestures = {
            'fist': 'Closed fist',
            'thumbs_up': 'Thumbs up',
            'thumbs_down': 'Thumbs down',
            'palm': 'Open palm',
            'pointing': 'Pointing finger',
            'peace': 'Peace sign',
            'rock': 'Rock sign',
            'ok': 'OK sign',
            'pinch': 'Pinch gesture'
        }

    def find_hands(self, img, draw=True):
        """Detect hands and draw landmarks if enabled"""
        if img is None:
            return None, None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.hand_type = []

        if self.results.multi_hand_landmarks:
            # Get hand type (left/right)
            if self.results.multi_handedness:
                for hand in self.results.multi_handedness:
                    self.hand_type.append(hand.classification[0].label)
            
            # Draw landmarks
            for i, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    
                    # Add hand type label (Left/Right)
                    if i < len(self.hand_type):
                        h, w, c = img.shape
                        wrist_x = int(hand_landmarks.landmark[0].x * w)
                        wrist_y = int(hand_landmarks.landmark[0].y * h)
                        cv2.putText(img, self.hand_type[i], (wrist_x, wrist_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img, self.results

    def find_position(self, img, hand_no=0):
        """Find the positions of hand landmarks for specified hand"""
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, landmark in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
                    
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """Determine which fingers are up (extended)"""
        fingers = []
        
        if not landmark_list or len(landmark_list) != 21:
            return []
        
        # Get hand type if available
        hand_type = self.hand_type[0] if len(self.hand_type) > 0 else "Right"
        
        # Thumb: Check horizontal position relative to the thumb base
        # Different logic for left and right hands
        if hand_type == "Right":
            # For right hand, thumb is up if tip is to the right of the thumb IP joint
            fingers.append(1 if landmark_list[4][1] > landmark_list[3][1] else 0)
        else:
            # For left hand, thumb is up if tip is to the left of the thumb IP joint
            fingers.append(1 if landmark_list[4][1] < landmark_list[3][1] else 0)
        
        # 4 Fingers: Check if fingertip is above finger PIP joint (knuckle)
        for id in range(1, 5):
            fingers.append(1 if landmark_list[self.tip_ids[id]][2] < landmark_list[self.tip_ids[id] - 2][2] else 0)
                
        return fingers
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
    def recognize_gesture(self, img, landmark_list):
        """Recognize hand gestures based on finger positions"""
        gesture = "Unknown"
        
        if not landmark_list or len(landmark_list) != 21:
            return img, gesture
        
        # Get fingers up status
        fingers = self.fingers_up(landmark_list)
        
        if not fingers:
            return img, gesture
        
        # Calculate distances for specific gesture detection
        thumb_tip = landmark_list[4]
        index_tip = landmark_list[8]
        middle_tip = landmark_list[12]
        ring_tip = landmark_list[16]
        pinky_tip = landmark_list[20]
        
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Recognize gestures
        # Fist (all fingers down)
        if sum(fingers) == 0:
            gesture = self.gestures['fist']
        
        # Open palm (all fingers up)
        elif sum(fingers) == 5:
            gesture = self.gestures['palm']
        
        # Pointing (only index finger up)
        elif fingers == [0, 1, 0, 0, 0]:
            gesture = self.gestures['pointing']
        
        # Peace sign (index and middle up)
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = self.gestures['peace']
        
        # Rock sign (index and pinky up, others down)
        elif fingers == [0, 1, 0, 0, 1]:
            gesture = self.gestures['rock']
        
        # Thumbs up (only thumb up, vertical position)
        elif fingers[0] == 1 and sum(fingers[1:]) == 0:
            # Check if thumb is pointing up
            if landmark_list[4][2] < landmark_list[9][2]:  # thumb tip above middle finger MCP
                gesture = self.gestures['thumbs_up']
            # Check if thumb is pointing down
            elif landmark_list[4][2] > landmark_list[9][2]:  # thumb tip below middle finger MCP
                gesture = self.gestures['thumbs_down']
        
        # OK sign (thumb and index form a circle, other fingers up)
        elif thumb_index_distance < 30 and sum(fingers[2:]) >= 2:
            gesture = self.gestures['ok']
        
        # Pinch gesture (thumb and index close, others can be in any position)
        elif thumb_index_distance < 30:
            gesture = self.gestures['pinch']
        
        # Draw gesture name on image
        cv2.putText(img, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img, gesture

def detect_from_image(image_path):
    """Run gesture detection on a static image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        detector = HandGestureRecognizer(mode=True)  # Static image mode
        img, _ = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        
        if landmark_list:
            img, gesture = detector.recognize_gesture(img, landmark_list)
            print(f"Detected gesture: {gesture}")
        else:
            print("No hand detected in the image")
        
        # Display result
        cv2.imshow("Image Gesture Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def main():
    """Main function to run gesture recognition on webcam or video feed"""
    # Check if an image path was provided
    if len(sys.argv) > 1:
        detect_from_image(sys.argv[1])
        return
    
    # Determine if running in WSL
    is_wsl = 'microsoft-standard' in platform.uname().release.lower()
    
    # Try different camera indices
    camera_indices = [0, 1, 2, '/dev/video0', '/dev/video1'] if is_wsl else [0]
    
    cap = None
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Successfully opened camera with index {idx}")
                break
        except Exception as e:
            print(f"Failed to open camera with index {idx}: {str(e)}")
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera.")
        print("Alternatives:")
        print("1. If you have a video file, enter file path below:")
        video_path = input("Video path (or press Enter to exit): ")
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video file: {video_path}")
                return
        else:
            print("Exiting program.")
            return
    
    # Initialize detector
    detector = HandGestureRecognizer()
    prev_time = 0
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame.")
                # If it's a video file, we've reached the end
                if not cap.isOpened() or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of video file.")
                    break
                continue
            
            # Find hands
            img, _ = detector.find_hands(img)
            landmark_list = detector.find_position(img)
            
            # Recognize gesture if hand detected
            if landmark_list:
                img, gesture = detector.recognize_gesture(img, landmark_list)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Show image
            cv2.imshow("Gesture Recognition", img)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()