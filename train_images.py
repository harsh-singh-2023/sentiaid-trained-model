import tensorflow as tf
import os
import json

# --- 1. Define Constants ---
DATA_DIR = 'asl_alphabet_train'
IMG_SIZE = (64, 64) # We will resize all images to 64x64
BATCH_SIZE = 32
NUM_CLASSES = 29 # 26 letters + space, del, nothing

# --- 2. Load Data ---
# 'image_dataset_from_directory' is a powerful function.
# It automatically finds all the sub-folders (A, B, C...)
# and labels the images for us.
print("Loading data...")

# Create the main dataset
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,  # Use 20% of the data for testing
    subset="both",         # Load both training and validation sets
    seed=42,               # 'seed' makes the random split reproducible
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

train_dataset, val_dataset = full_dataset

# Get the class names (A, B, C, ...)
class_names = train_dataset.class_names
print(f"Found {len(class_names)} classes: {class_names}")

# --- 3. Build the CNN Model ---
print("Building model...")

model = tf.keras.models.Sequential([
    # This layer rescales the pixel values from [0, 255] to [0, 1]
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # First Convolutional block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second Convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third Convolutional block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten the 3D features into a 1D vector
    tf.keras.layers.Flatten(),
    
    # A standard Dense (fully-connected) layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Dropout helps prevent overfitting
    
    # Output layer. 'NUM_CLASSES' is 29.
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this loss for integer labels
    metrics=['accuracy']
)

model.summary()

# --- 4. Train the Model ---
print("Training model...")

# 'epochs=10' is a good start. This will take a few minutes.
# On a real project, you'd run for more epochs.
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10 
)

# --- 5. Save the New, Smart Model ---
print("Saving model...")

# Save in the new Keras format
model.save('asl_alphabet_classifier.keras')

# Save the labels (A, B, C...) for our API
config = {"labels": class_names}
with open('asl_alphabet_config.json', 'w') as f:
    json.dump(config, f)

print("Model saved as 'asl_alphabet_classifier.keras'")
print("Config saved as 'asl_alphabet_config.json'")

# --- 6. Evaluate the Model ---
print("\nEvaluating model on test data...")
score = model.evaluate(val_dataset, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')
