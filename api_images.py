import uvicorn
import numpy as np
import tensorflow as tf
import json  # Make sure this is imported
import io    # Make sure this is imported
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# --- 2. Add CORS Middleware ---
# This allows your index.html to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Load Your Trained Model and Config ---
print("Loading model and config...")
MODEL_PATH = "asl_alphabet_classifier.keras"
CONFIG_PATH = "asl_alphabet_config.json"

# Check if files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
    print("--------------------------------------------------")
    print(f"ERROR: Could not find model or config files.")
    print("Did you run 'python train_images.py' first?")
    print("--------------------------------------------------")
    exit()

# Load the model you trained
model = tf.keras.models.load_model(MODEL_PATH)

# Load the config file
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

LABELS = config['labels'] # This is your list of ["A", "B", "C"...]

# --- THIS IS THE FIX ---
# Instead of hard-coding (64, 64), we read it from the config.
IMAGE_SIZE = (64, 64)
# --- END OF FIX ---

print(f"Model and config loaded. Ready to predict on {len(LABELS)} classes.")
print(f"Expecting image size: {IMAGE_SIZE}")


# --- 4. Helper Function to Preprocess Image ---
def preprocess_image(image_bytes):
    """
    Takes image bytes from the API, opens them with PIL,
    and preprocesses them to match the model's input.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case it's PNG with an alpha channel)
    image = image.convert('RGB')
    
    # Resize to the model's expected input size
    image = image.resize(IMAGE_SIZE)
    
    # Convert the PIL image to a NumPy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # The model was trained on batches of images.
    # We need to add an extra dimension to represent the "batch",
    # so (64, 64, 3) becomes (1, 64, 64, 3).
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# --- 5. Define Your API Prediction Endpoint ---
@app.post("/predict_image")
async def predict_sign(image_file: UploadFile = File(...)):
    
    try:
        # Read the image as bytes
        image_bytes = await image_file.read()
        
        # --- Step 1: Preprocess Image ---
        processed_image = preprocess_image(image_bytes)
        
        # --- Step 2: Make Prediction ---
        prediction_array = model.predict(processed_image)
        
        # --- Step 3: Decode Prediction ---
        predicted_index = np.argmax(prediction_array[0])
        predicted_letter = LABELS[predicted_index]
        confidence = float(prediction_array[0][predicted_index])
        
        print(f"Prediction: {predicted_letter}, Confidence: {confidence:.4f}")
        
        # Return the result as JSON
        return {
            "predicted_letter": predicted_letter,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

# --- 6. Add a "Root" Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the ASL Alphabet Recognition API!"}

# --- 7. (Optional) Run the app when script is executed ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
