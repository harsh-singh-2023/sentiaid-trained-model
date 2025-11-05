import cv2
import mediapipe as mp

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- Webcam Setup ---
# 0 is the default webcam
cap = cv2.VideoCapture(0)

# Initialize Holistic model
# min_detection_confidence: How confident it needs to be to detect a person
# min_tracking_confidence: How confident it needs to be to track the person
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        # Read frame from webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- MediaPipe Processing ---
        # 1. To improve performance, mark the image as not writeable
        image.flags.writeable = False
        
        # 2. Convert from BGR (OpenCV's format) to RGB (MediaPipe's format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Process the image and find all landmarks
        results = holistic.process(image)

        # --- Drawing Landmarks ---
        # 4. Revert to BGR for OpenCV to be able to draw on it
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 5. Draw Face landmarks (NMMs!)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )

        # 6. Draw Pose (Body) landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

        # 7. Draw Left Hand landmarks
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

        # 8. Draw Right Hand landmarks
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # --- Display the final image ---
        cv2.imshow('MediaPipe Holistic - Press Q to Quit', image)

        # Press 'q' to quit the window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release all resources
cap.release()
cv2.destroyAllWindows()