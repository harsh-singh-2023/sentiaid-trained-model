import uvicorn
import numpy as np
import os
import cv2
import mediapipe as mp
import tempfile
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# --- 2. Load Your Trained Model and Define Constants ---
# Load the model you saved in Phase 2
MODEL_PATH = "sign_recognizer.h5"
model = load_model(MODEL_PATH)

# These must be the *same* as in your train.py
LABELS = ["hello", "thanks", "goodbye"]
# This was the max length from the dummy data.
# When you train on WLASL, you'll update this.
MAX_LENGTH = 100 

# --- 3. Copy Your Landmark Extraction Function from Phase 1 ---
# We need this function to process the new videos.
mp_holistic = mp.solutions.holistic
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

def extract_keypoints(video_path):
    """
    Processes a single video file and returns a NumPy array 
    of its keypoints.
    """
    all_frame_data = []
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Extract landmarks and flatten
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            ).flatten() if results.pose_landmarks else np.zeros(NUM_POSE_LANDMARKS * 4)
                
            face = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
            ).flatten() if results.face_landmarks else np.zeros(NUM_FACE_LANDMARKS * 3)
                
            lh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten() if results.left_hand_landmarks else np.zeros(NUM_HAND_LANDMARKS * 3)
                
            rh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten() if results.right_hand_landmarks else np.zeros(NUM_HAND_LANDMARKS * 3)
                
            frame_keypoints = np.concatenate([pose, face, lh, rh])
            all_frame_data.append(frame_keypoints)
            
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(all_frame_data)


# --- 4. Define Your API Prediction Endpoint ---
@app.post("/predict")
async def predict_sign(video: UploadFile = File(...)):
    
    # FastAPI saves the uploaded file to a temporary location
    # We must use a 'tempfile' to get a path that OpenCV can read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(await video.read())
        video_path = tmp.name
    
    try:
        # --- Step 1: Process Video (Phase 1) ---
        print(f"Processing video at: {video_path}")
        keypoints = extract_keypoints(video_path)
        
        if keypoints.shape[0] == 0:
            return {"error": "Could not process video. No frames found."}
        
        # --- Step 2: Preprocess Data (Phase 2) ---
        # 1. We have a sequence of (num_frames, 1662)
        # We need to wrap it in a list to make it a "batch" of 1
        sequence_batch = [keypoints] 
        
        # 2. Pad the sequence to the same length the model was trained on
        padded_sequence = pad_sequences(
            sequence_batch, 
            dtype='float32', 
            padding='pre', 
            maxlen=MAX_LENGTH, 
            value=0.0
        )
        
        # --- Step 3: Make Prediction ---
        # padded_sequence shape is now (1, 100, 1662), which is what the model expects
        prediction_array = model.predict(padded_sequence)
        
        # --- Step 4: Decode Prediction ---
        # 'prediction_array' looks like: [[0.1, 0.8, 0.1]]
        # 'np.argmax' finds the index of the highest value (e.g., index 1)
        predicted_index = np.argmax(prediction_array[0])
        
        # Use the index to get the word from our LABELS list
        predicted_word = LABELS[predicted_index]
        
        # Get the confidence score
        confidence = float(prediction_array[0][predicted_index])
        
        print(f"Prediction: {predicted_word}, Confidence: {confidence:.4f}")
        
        # Return the result as JSON
        return {
            "predicted_word": predicted_word,
            "confidence": confidence
        }
        
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)

# --- 5. Add a "Root" Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language AI API!"}

# --- 6. (Optional) Run the app when script is executed ---
# This line allows you to run: python api.py
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)