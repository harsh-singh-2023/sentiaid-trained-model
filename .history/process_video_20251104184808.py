import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic

# Define the number of landmarks to avoid errors
# Pose (33 landmarks * 4 values: x,y,z,visibility) = 132
# Face (468 landmarks * 3 values: x,y,z) = 1404
# Left Hand (21 landmarks * 3 values) = 63
# Right Hand (21 landmarks * 3 values) = 63
# TOTAL = 132 + 1404 + 63 + 63 = 1662 values per frame
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

def extract_keypoints(video_path):
    """
    Processes a single video file and returns a NumPy array 
    of its keypoints.
    """
    
    # This list will hold the keypoint data for every frame
    all_frame_data = []
    
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(
        static_image_mode=False, # Use for video
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # Video has ended
                break
                
            # Convert color from BGR (OpenCV) to RGB (MediaPipe)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Mark as not writeable to speed up
            
            # Process the image to get landmarks
            results = holistic.process(image)
            
            # --- This is the new, important part ---
            # Instead of drawing, we extract and save the data.
            
            # 1. Pose Landmarks
            if results.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                ).flatten() # Flatten to 1D array (33*4 = 132 values)
            else:
                # If no pose found, create an empty array of zeros
                pose = np.zeros(NUM_POSE_LANDMARKS * 4) 
                
            # 2. Face Landmarks (NMMs!)
            if results.face_landmarks:
                face = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
                ).flatten() # Flatten to 1D array (468*3 = 1404 values)
            else:
                face = np.zeros(NUM_FACE_LANDMARKS * 3)
                
            # 3. Left Hand Landmarks
            if results.left_hand_landmarks:
                lh = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                ).flatten() # Flatten to 1D array (21*3 = 63 values)
            else:
                lh = np.zeros(NUM_HAND_LANDMARKS * 3)
                
            # 4. Right Hand Landmarks
            if results.right_hand_landmarks:
                rh = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                ).flatten() # Flatten to 1D array (21*3 = 63 values)
            else:
                rh = np.zeros(NUM_HAND_LANDMARKS * 3)
                
            # 5. Concatenate all landmarks into a single row
            # This creates one long vector of 1662 numbers for this frame
            frame_keypoints = np.concatenate([pose, face, lh, rh])
            
            # 6. Add this frame's data to our list
            all_frame_data.append(frame_keypoints)
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert the list of arrays into a single 2D NumPy array
    # Shape will be: (num_frames, 1662)
    return np.array(all_frame_data)


# --- This is the main part that runs your script ---
if __name__ == "__main__":
    
    # 1. DEFINE YOUR VIDEO
    # IMPORTANT: Change this to the name of a video file in your project folder
    VIDEO_NAME = "my_test_video.mp4" 
    VIDEO_PATH = os.path.join(".", VIDEO_NAME) # Gets the full path

    # 2. DEFINE YOUR OUTPUT
    OUTPUT_NAME = "my_test_data.npy"
    OUTPUT_PATH = os.path.join(".", OUTPUT_NAME)
    
    # 3. PROCESS THE VIDEO
    print(f"Starting processing for: {VIDEO_NAME}...")
    keypoints = extract_keypoints(VIDEO_PATH)
    
    # 4. SAVE THE DATA
    np.save(OUTPUT_PATH, keypoints)
    
    print("---" * 10)
    print(f"Successfully processed video!")
    print(f"Data saved to: {OUTPUT_NAME}")
    print(f"Data shape: {keypoints.shape}")
    print(f"(This means {keypoints.shape[0]} frames, with {keypoints.shape[1]} data points each)")
    print("---" * 10)