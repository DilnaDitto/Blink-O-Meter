# (Python - VS Code)
# Blink-o-Meter: Confidence Through the Eyes
# Requires: OpenCV, MediaPipe, Numpy

import cv2
import mediapipe as mp
import time
import random
import numpy as np

# --- Constants ---
# MediaPipe landmark indices for Eye Aspect Ratio (EAR) calculation.
# The vertical points are crucial for measuring the opening.
LEFT_EYE_LANDMARKS = [33, 159, 145, 133]  # [left_corner, top, bottom, right_corner]
RIGHT_EYE_LANDMARKS = [362, 386, 374, 263] # [left_corner, top, bottom, right_corner]

# Blink detection threshold and state
EYE_ASPECT_RATIO_THRESHOLD = 0.23

# Timing
BLINK_RATE_CALCULATION_PERIOD_S = 60  # seconds

# Confidence calculation
MYSTERY_FACTOR_MIN = 2
MYSTERY_FACTOR_MAX = 5

# --- UI & Display ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIDENCE_POS = (30, 50)
BLINK_RATE_POS = (30, 100)
MESSAGE_POS = (30, 150)
CONFIDENCE_COLOR = (0, 255, 0)
BLINK_RATE_COLOR = (255, 255, 0)
MESSAGE_COLOR = (255, 0, 255)

# --- Helper Functions ---

def euclidean_dist(pt1, pt2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def get_eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.
    EAR is the ratio of the vertical distance between eyelids to the
    horizontal distance between the eye corners.
    """
    p_hor_start = landmarks[eye_indices[0]]
    p_ver_top = landmarks[eye_indices[1]]
    p_ver_bottom = landmarks[eye_indices[2]]
    p_hor_end = landmarks[eye_indices[3]]

    ver_dist = euclidean_dist(p_ver_top, p_ver_bottom)
    hor_dist = euclidean_dist(p_hor_start, p_hor_end)

    # Avoid division by zero if eye is not detected
    if hor_dist == 0:
        return 0.0

    return ver_dist / hor_dist

def get_confidence_message(confidence):
    """Returns a fun message based on the confidence score."""
    if confidence > 80:
        return "Movie villain level calm 😈"
    elif confidence > 50:
        return "Probably fine 🙂"
    else:
        return "Either stressed or cutting onions 🧅"

def main():
    """Main function to run the Blink-o-Meter application."""
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh

    blink_count = 0
    blink_rate = 0
    start_time = time.time()
    eye_closed = False # State variable to track if eye is closed
    mystery_factor = random.randint(MYSTERY_FACTOR_MIN, MYSTERY_FACTOR_MAX)
    
    # Use a 'with' block for robust resource management
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame for a mirror-like view and improve performance
            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False # Mark as not writeable to pass by reference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            frame.flags.writeable = True # Mark as writeable again for drawing

            if results.multi_face_landmarks:
                # Assuming only one face
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

                left_ear = get_eye_aspect_ratio(points, LEFT_EYE_LANDMARKS)
                right_ear = get_eye_aspect_ratio(points, RIGHT_EYE_LANDMARKS)
                ear = (left_ear + right_ear) / 2.0

                # Stateful blink detection
                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    if not eye_closed:
                        blink_count += 1
                        eye_closed = True # Mark eye as closed
                else:
                    eye_closed = False # Mark eye as open

            # Calculate blink rate every minute
            elapsed_time = time.time() - start_time
            if elapsed_time >= BLINK_RATE_CALCULATION_PERIOD_S:
                blink_rate = blink_count
                blink_count = 0
                start_time = time.time()
                mystery_factor = random.randint(MYSTERY_FACTOR_MIN, MYSTERY_FACTOR_MAX)

            confidence = max(0, 100 - (blink_rate * mystery_factor))
            msg = get_confidence_message(confidence)

            # Display info on the frame
            cv2.putText(frame, f"Confidence: {confidence}%", CONFIDENCE_POS, FONT, 1.2, CONFIDENCE_COLOR, 3)
            cv2.putText(frame, f"Blinks/min: {blink_rate}", BLINK_RATE_POS, FONT, 1, BLINK_RATE_COLOR, 2)
            cv2.putText(frame, msg, MESSAGE_POS, FONT, 0.8, MESSAGE_COLOR, 2)

            cv2.imshow("Blink-o-Meter", frame)

            # Exit on 'ESC' key
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()