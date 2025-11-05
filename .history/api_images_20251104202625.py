import uvicorn
import numpy as np
import tensorflow as tf
import json
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # Import this
from PIL import Image

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# --- 2. Add CORS Middleware ---
# This is a CRITICAL step for the frontend to work.
# It allows your browser (from index.html) to talk to this API (at port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Load Your Trained Model and Config ---
print("Loading model and config...")
# Load the model you trained
MODEL_PATH = "Unvoiced/asl_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load the config file

IMAGE_SIZE = (64, 64) # Must match the IMG_SIZE in train_images.py
print(f"Model and config loaded. Ready to predict on {len(LABELS)} classes.")

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
    
    # Rescaling was the first layer in the model, 
    # but we can also do it here.
    # model.predict() is often faster if the input is already rescaled.
    # We'll let the model's Rescaling layer handle it.
    
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
        # 'prediction_array' looks like: [[0.01, 0.05, 0.9, ...]]
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
        return {"error": f"An error occurred: {str(e)}"}

# --- 6. Add a "Root" Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the ASL Alphabet Recognition API!"}

# --- 7. (Optional) Run the app when script is executed ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
