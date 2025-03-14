import cv2
import mediapipe as mp
import time
import sys
import platform

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        if img is None:
            return None, None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        
        return img, self.results

    def find_position(self, img, hand_no=0):
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = img.shape
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

    detector = HandDetector()
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

            # Find hands
            img, results = detector.find_hands(img)
            landmark_list = detector.find_position(img)

            # Display FPS
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If landmarks detected, show info
            if len(landmark_list) > 0:
                print(f"Wrist position: {landmark_list[0]}")

            cv2.imshow("Hand Detection", img)
            
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