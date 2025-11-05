import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --- 1. Data Simulation ---
# We don't have the full WLASL dataset yet.
# Let's create a "dummy" dataset to prove our code works.
# This simulates 100 "videos" for 3 words: "hello", "thanks", "goodbye".
# Each video has 1662 data points per frame.

def create_dummy_data(base_path="data", num_samples=100, num_classes=3):
    print("Creating dummy data...")
    labels = ["hello", "thanks", "goodbye"]
    data_dir = os.path.join(base_path)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for i in range(num_samples):
        # Pick a random word
        label = np.random.choice(labels)
        
        # Create a video of random length (e.g., 50 to 100 frames)
        seq_len = np.random.randint(50, 101)
        
        # Create random keypoint data
        # This is what your process_video.py creates
        dummy_video = np.random.rand(seq_len, 1662) 
        
        # Save the data
        # We'll save as: data/hello_1.npy, data/hello_2.npy etc.
        count = len([f for f in os.listdir(data_dir) if f.startswith(label)])
        file_path = os.path.join(data_dir, f"{label}_{count}.npy")
        np.save(file_path, dummy_video)
        
    print(f"Created {num_samples} dummy .npy files in '{data_dir}' folder.")
    return data_dir, labels

# --- 2. Load and Preprocess Data ---

def load_data(data_dir, labels):
    sequences = [] # This will be our X (the video data)
    word_labels = [] # This will be our y (the words)
    
    for label in labels:
        files = [f for f in os.listdir(data_dir) if f.startswith(label)]
        for file in files:
            # Load the .npy file
            video_data = np.load(os.path.join(data_dir, file))
            sequences.append(video_data)
            word_labels.append(label)
            
    # --- Preprocessing ---
    
    # 1. Pad sequences
    # Videos have different lengths (e.g., 76 frames, 90 frames, etc.)
    # The AI needs them to be the *same length*.
    # 'pad_sequences' will add zeros to the end of shorter videos.
    # 'pre' padding is often better for LSTMs.
    X = pad_sequences(sequences, dtype='float32', padding='pre', value=0.0)
    
    # 2. Encode labels
    # Convert text ("hello", "thanks") to numbers (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(word_labels)
    
    # 3. One-Hot Encode labels
    # Convert numbers (0, 1, 2) to "one-hot" vectors:
    # 0 -> [1, 0, 0]
    # 1 -> [0, 1, 0]
    # 2 -> [0, 0, 1]
    # This is what the model's output layer will predict.
    y = to_categorical(y_encoded)
    
    # 'X' is our video data, 'y' is our labels
    # 'le' is the label encoder we'll need later to decode predictions
    return X, y, le

# --- 3. Build the LSTM Model ---
# This is the "brain" of the AI

def build_model(num_classes, input_shape):
    model = Sequential()
    
    # Masking layer: Ignores the '0.0' padding we added
    # This is VERY important for LSTMs.
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    # LSTM layer: This is where the magic happens
    # It learns patterns *over time* (through the frames)
    model.add(LSTM(64, return_sequences=False)) # 64 "neurons"
    
    # Output layer: A standard "Dense" layer
    # 'softmax' ensures the output is a probability (e.g., 90% "hello")
    model.add(Dense(num_classes, activation='softmax'))
    
    # Put it all together
    # 'categorical_crossentropy' is the standard loss for multi-class classification
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --- 4. Main Training Script ---
if __name__ == "__main__":
    
    # 1. Create our dummy data (you'll skip this when you have real data)
    DATA_PATH, LABELS = create_dummy_data()
    
    # 2. Load and process the data
    print("Loading and preprocessing data...")
    X, y, label_encoder = load_data(DATA_PATH, LABELS)
    
    # 3. Split data into training and testing sets
    # Train on 80%, test on 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Total samples: {len(X)}")
    print(f"Training on: {len(X_train)} samples")
    print(f"Testing on: {len(X_test)} samples")
    
    # 4. Build the model
    # Get the shape from our padded data: (num_videos, max_frames, num_keypoints)
    # e.g., (100, 100, 1662)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(LABELS)
    
    print("Building model...")
    model = build_model(num_classes=num_classes, input_shape=input_shape)
    model.summary()
    
    # 5. Train the model
    print("Training model...")
    # 'epochs' is how many times to "see" the data. 
    # This will be very fast on dummy data.
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    
    # 6. Save the model
    # We save the model file to be used by our API in Phase 3
    print("Saving model...")
    model.save('sign_recognizer.h5')
    print("Model saved as 'sign_recognizer.h5'")
    
    # 7. (Optional) Test the model
    print("Testing model on a test sample...")
    prediction = model.predict(np.expand_dims(X_test[0], axis=0))
    predicted_index = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([predicted_index])[0]
    
    actual_index = np.argmax(y_test[0])
    actual_word = label_encoder.inverse_transform([actual_index])[0]
    
    print(f"Actual Word: {actual_word}")
    print(f"Predicted Word: {predicted_word}")