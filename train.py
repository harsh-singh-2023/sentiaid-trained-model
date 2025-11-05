import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --- 1. Define Constants ---
# Use the real data path
DATA_PATH = os.path.join("WLASL_data") 

# --- 2. Load and Preprocess Data ---

def load_data(data_dir):
    sequences = [] # This will be our X (the video data)
    word_labels = [] # This will be our y (the words)
    
    # Get all the subdirectories (which are the labels)
    labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(labels)} words (classes) in dataset.")
    if not labels:
        print(f"Error: No data found in '{data_dir}'.")
        print("Please run 'python build_dataset.py' first.")
        exit()
        
    for label in labels:
        label_path = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
        
        for file in files:
            # Load the .npy file
            video_data = np.load(os.path.join(label_path, file))
            sequences.append(video_data)
            word_labels.append(label)
            
    print(f"Loaded {len(sequences)} video samples.")
            
    # --- Preprocessing ---
    
    # 1. Pad sequences
    # We pad to the longest video in our dataset
    # 'pre' padding is often better for LSTMs
    X = pad_sequences(sequences, dtype='float32', padding='pre', value=0.0)
    
    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(word_labels)
    
    # 3. One-Hot Encode labels
    y = to_categorical(y_encoded)
    
    # Save the max length and labels for the API
    max_len = X.shape[1]
    
    return X, y, le, max_len, labels

# --- 3. Build the LSTM Model ---
# We make the model a bit "deeper" now for real data
def build_model(num_classes, input_shape):
    model = Sequential()
    
    # Masking layer: Ignores the '0.0' padding
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    
    # A "deeper" model with two LSTM layers
    # return_sequences=True tells the first LSTM to pass its
    # full sequence output to the next LSTM layer.
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2)) # Dropout helps prevent overfitting
    
    # The second LSTM layer. return_sequences=False is the default.
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    # A standard hidden layer
    model.add(Dense(64, activation='relu'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --- 4. Main Training Script ---
if __name__ == "__main__":
    
    # 1. Load and process the REAL data
    print("Loading and preprocessing REAL data...")
    X, y, label_encoder, MAX_LEN, LABELS = load_data(DATA_PATH)
    
    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Max video length (padded to): {MAX_LEN} frames")
    print(f"Training on: {len(X_train)} samples")
    print(f"Testing on: {len(X_test)} samples")
    
    # 3. Build the model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (MAX_LEN, 1662)
    num_classes = len(LABELS)
    
    print("Building model...")
    model = build_model(num_classes=num_classes, input_shape=input_shape)
    model.summary()
    
    # 4. Train the model
    print("Training model...")
    # We train for more epochs on real data
    # batch_size=16 is a good default for this size
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
    
    # 5. Save the new, SMART model
    print("Saving model...")
    # We save in Keras's native format now. It's better.
    model.save('sign_recognizer_WLASL.keras') 
    
    # We also save the 'config' (labels and max_len) for our API
    config = {
        "labels": list(label_encoder.classes_),
        "max_len": int(MAX_LEN)
    }
    with open('model_config.json', 'w') as f:
        json.dump(config, f)
        
    print("Model saved as 'sign_recognizer_WLASL.keras'")
    print("Config saved as 'model_config.json'")
    
    # 6. Test the model on a sample
    print("\nTesting model on a test sample...")
    prediction = model.predict(np.expand_dims(X_test[0], axis=0))
    predicted_index = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([predicted_index])[0]
    
    actual_index = np.argmax(y_test[0])
    actual_word = label_encoder.inverse_transform([actual_index])[0]
    
    print(f"Actual Word: {actual_word}")
    print(f"Predicted Word: {predicted_word}")
    
    # 7. Evaluate the model on the whole test set
    print("\nEvaluating model on all test data...")
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

