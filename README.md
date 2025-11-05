This project uses a computer vision model to recognize American Sign Language (ASL) alphabet signs in real-time from a webcam.

It is built with Python, TensorFlow/Keras, and a FastAPI backend. The frontend is a simple HTML/JavaScript page.

## How to Run

### 1. Prerequisites

* Python 3.11
* Git and Git LFS

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harsh-singh-2023/sentiaid-trained-model.git
    cd ASL-Alphabet-Recognizer
    ```

2.  **Create a virtual environment:**
    ```bash
    py -3.11 -m venv sign_env
    .\sign_env\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset (One-Time Setup):**
    * This project uses the [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
    * Download the `archive.zip` file.
    * Unzip it and place the `asl_alphabet_train` folder inside this project directory.

### 3. Train Your Own Model (Optional)

You can train your own model on the dataset by running:

```bash
python train_images.py
```

This will create a new `asl_alphabet_classifier.keras` file.

### 4. Run the Application

1.  **Start the API Server:**
    ```bash
    # This will load the pre-trained model
    uvicorn api_images:app --reload
    ```

2.  **Open the Frontend:**
    * In your file explorer, double-click the `index.html` file.
    * Allow camera permissions in your browser.

3.  Hold up an ASL sign and click "Predict"!
