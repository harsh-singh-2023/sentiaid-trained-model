import os
import json
import numpy as np
import yt_dlp
import shutil
from process_video import extract_keypoints # Import your function!

# --- 1. Define Constants ---

# Path to the JSON file from the WLASL repo you cloned
WLASL_JSON_PATH = os.path.join("WLASL", "start_kit", "WLASL_v0.3.json")

# Create a new folder to hold our processed .npy files
DATA_PATH = os.path.join("WLASL_data") 

# Create a temporary folder to store video clips
TEMP_PATH = os.path.join("videos_temp")

# --- We will start small! ---
# Process the first 10 words
MAX_WORDS_TO_PROCESS = 10 
# And only the first 5 videos for each word
MAX_VIDEOS_PER_WORD = 5 

FPS = 25 # WLASL videos are 25 FPS

# --- 2. Helper Functions ---

def load_wlasl_json(json_path):
    """Loads the WLASL JSON file."""
    print(f"Loading JSON from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def download_and_clip(url, video_id, frame_start, frame_end):
    """
    Downloads a video, clips it using ffmpeg, and saves it.
    Returns the path to the clipped video.
    """
    
    # Calculate start time and duration in seconds
    start_time = (frame_start - 1) / FPS
    end_time = (frame_end - 1) / FPS
    duration = end_time - start_time
    
    # Define the output path for the temporary clip
    temp_video_path = os.path.join(TEMP_PATH, f"{video_id}.mp4")
    
    # yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': temp_video_path,
        'quiet': True,
        'overwrites': True,
        'nocheckcertificate': True,
        'postprocessor_args': [
            '-ss', str(start_time),  # Start time
            '-t', str(duration),     # Duration
            '-c:v', 'libx264',       # Re-encode video
            '-c:a', 'aac'            # Re-encode audio
        ]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if os.path.exists(temp_video_path):
            return temp_video_path
        else:
            return None
    except Exception as e:
        print(f"  yt-dlp Error: {e}")
        return None

# --- 3. Main Script Logic ---

if __name__ == "__main__":
    
    # Create the output directories if they don't exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    
    # Load the "map" of all videos
    wlasl_data = load_wlasl_json(WLASL_JSON_PATH)
    if not wlasl_data:
        print("Could not load WLASL JSON. Exiting.")
        exit()
        
    print(f"Starting dataset build for {MAX_WORDS_TO_PROCESS} words...")
    
    word_counter = 0
    total_videos_processed = 0
    
    # Loop over each word (gloss) in the dataset
    for entry in wlasl_data:
        if word_counter >= MAX_WORDS_TO_PROCESS:
            break
            
        gloss = entry['gloss']
        print(f"\nProcessing word ({word_counter + 1}/{MAX_WORDS_TO_PROCESS}): '{gloss}'")
        
        # Create a folder for this word's .npy files
        gloss_path = os.path.join(DATA_PATH, gloss)
        os.makedirs(gloss_path, exist_ok=True)
        
        video_counter = 0
        # Loop over each video instance for that word
        for instance in entry['instances']:
            if video_counter >= MAX_VIDEOS_PER_WORD:
                break
            
            url = instance['url']
            video_id = instance['video_id']
            frame_start = instance['frame_start']
            frame_end = instance['frame_end']

            # 1. Download and Clip the video
            print(f"  Downloading video {video_counter + 1}/{MAX_VIDEOS_PER_WORD} (ID: {video_id})...", end="", flush=True)
            clipped_video_path = download_and_clip(url, video_id, frame_start, frame_end)
            
            if clipped_video_path:
                # 2. Process the clip into keypoints (using your Phase 1 script)
                keypoints = extract_keypoints(clipped_video_path)
                
                if keypoints.shape[0] > 0:
                    # 3. Save the .npy file
                    npy_path = os.path.join(gloss_path, f"{video_id}.npy")
                    np.save(npy_path, keypoints)
                    print(f" Done. (Saved {keypoints.shape[0]} frames)")
                    video_counter += 1
                    total_videos_processed += 1
                else:
                    print(" Failed. (Could not extract keypoints)")
            else:
                print(" Failed. (Download error)")
                
        word_counter += 1
        
    # 4. Clean up
    print(f"\nCleaning up temporary video folder: {TEMP_PATH}")
    shutil.rmtree(TEMP_PATH)
    
    print("\n---" * 10)
    print("Dataset build complete!")
    print(f"Total words processed: {word_counter}")
    print(f"Total videos saved: {total_videos_processed}")
    print(f"Your real data is now in the '{DATA_PATH}' folder.")
    print("---" * 10)
