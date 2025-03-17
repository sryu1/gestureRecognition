import cv2
import mediapipe as mp
import time
import sys
import platform
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class GestureRecognizer:
    def __init__(self, model_path='gesture_recognizer.task', max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.max_hands = max_hands
        
        # Initialize MediaPipe Gesture Recognizer
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence
        )
        
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
    def process_image(self, img, draw=True):
        """Process image and recognize gestures"""
        if img is None:
            return None, None, None
        
        # Convert to RGB and create MediaPipe Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Process the image
        recognition_result = self.recognizer.recognize(mp_image)
        
        gestures = []
        hand_landmarks_list = []
        
        # Process recognition results
        if recognition_result.gestures and recognition_result.hand_landmarks:
            for idx, (gesture_list, hand_landmarks) in enumerate(zip(
                    recognition_result.gestures, recognition_result.hand_landmarks)):
                
                if not gesture_list:
                    continue
                    
                # Get top gesture
                top_gesture = gesture_list[0]
                gestures.append((top_gesture.category_name, top_gesture.score))
                hand_landmarks_list.append(hand_landmarks)
                
                if draw:
                    # Convert to drawing format
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z
                        ) for landmark in hand_landmarks
                    ])
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks_proto,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Draw gesture name
                    gesture_name = top_gesture.category_name
                    confidence = top_gesture.score
                    text = f"{gesture_name} ({confidence:.2f})"
                    
                    # Determine text position based on hand location
                    x_center = min(int(sum(landmark.x for landmark in hand_landmarks) * img.shape[1] / len(hand_landmarks)), img.shape[1] - 10)
                    y_center = min(int(sum(landmark.y for landmark in hand_landmarks) * img.shape[0] / len(hand_landmarks)), img.shape[0] - 10)
                    
                    cv2.putText(
                        img,
                        text,
                        (x_center - 50, y_center - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
        
        return img, gestures, hand_landmarks_list
    
    def find_position(self, hand_landmarks, img_shape):
        """Convert normalized landmarks to pixel coordinates"""
        height, width, _ = img_shape
        landmark_list = []
        
        for id, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            landmark_list.append([id, cx, cy])
            
        return landmark_list

def main():
    # Determine the operating system
    is_wsl = 'microsoft-standard' in platform.uname().release.lower()
    
    # Camera selection logic
    if is_wsl:
        print("WSL detected. Trying different camera indices...")
        camera_indices = [0, 1, 2, '/dev/video0', '/dev/video1', '/dev/video2']
    else:
        camera_indices = [0]  # For non-WSL systems, just try the default camera

    cap = None
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Successfully opened camera with index {idx}")
                break
        except Exception as e:
            print(f"Failed to open camera with index {idx}: {str(e)}")
            continue

    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera. Please check:")
        print("1. If you're using WSL, make sure you have WSL2 and USB camera support")
        print("2. Check if the camera is being used by another application")
        print("3. Verify camera permissions")
        print("4. Ensure the camera is properly connected")
        sys.exit(1)

    # Initialize the gesture recognizer
    try:
        model_path = 'gesture_recognizer.task'
        recognizer = GestureRecognizer(model_path=model_path)
        print(f"Initialized gesture recognizer with model: {model_path}")
    except Exception as e:
        print(f"Error initializing gesture recognizer: {str(e)}")
        print("Make sure the model file 'gesture_recognizer.task' is in the current directory")
        print("You can download it from: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task")
        sys.exit(1)
    
    prev_time = 0
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame. Retrying...")
                continue

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # Process image and recognize gestures
            img, gestures, hand_landmarks_list = recognizer.process_image(img)

            # Display FPS
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display number of hands detected
            cv2.putText(img, f"Hands: {len(hand_landmarks_list)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display recognized gestures
            for i, (gesture_name, confidence) in enumerate(gestures):
                # Find position for each hand
                if hand_landmarks_list and i < len(hand_landmarks_list):
                    landmarks = recognizer.find_position(hand_landmarks_list[i], img.shape)
                    if landmarks and len(landmarks) > 0:
                        # Print wrist position (landmark 0)
                        print(f"Hand {i} - Gesture: {gesture_name}, Wrist position: {landmarks[0]}")

            cv2.imshow("Gesture Recognition", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()