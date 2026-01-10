# Real-Time Emotion Detection using DeepFace and OpenCV
# Multi-Face Real-Time Emotion Detection with CSV Logging
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
from collections import deque
import time
import csv
import os

# ------------------------------
# CSV Logging Setup
# ------------------------------
csv_file = "emotion_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "person_id", "emotion"])  # headers

# ------------------------------
# Initialize Webcam & Detector
# ------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = MTCNN()

# For temporal smoothing
emotion_history = {}  # {person_id: deque()}
frame_interval = 1  # seconds
last_update = time.time()

# Simple ID tracker
next_person_id = 1
face_positions = {}  # {person_id: (x_center, y_center)}

# Distance threshold to match faces to existing IDs
MATCH_THRESHOLD = 50

# ------------------------------
# Helper Function: Find closest face ID
# ------------------------------
def get_person_id(x, y):
    global next_person_id, face_positions
    for pid, (px, py) in face_positions.items():
        if abs(px - x) < MATCH_THRESHOLD and abs(py - y) < MATCH_THRESHOLD:
            return pid
    # New person
    pid = next_person_id
    next_person_id += 1
    return pid

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    current_face_positions = {}

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        x_center = x + w // 2
        y_center = y + h // 2

        # Assign ID
        person_id = get_person_id(x_center, y_center)
        current_face_positions[person_id] = (x_center, y_center)

        face_crop = rgb_frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend='mtcnn'
            )

            dominant_emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][dominant_emotion]

            # Initialize deque if first time
            if person_id not in emotion_history:
                emotion_history[person_id] = deque(maxlen=10)

            # Update emotion if confidence high
            if confidence > 50:
                emotion_history[person_id].append(dominant_emotion)

                # Log to CSV
                timestamp = time.time()
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, person_id, dominant_emotion])

            # Smoothed emotion
            if len(emotion_history[person_id]) > 0:
                smoothed_emotion = max(set(emotion_history[person_id]),
                                       key=emotion_history[person_id].count)
            else:
                smoothed_emotion = "Detecting..."

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_id}: {smoothed_emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except:
            pass

    # Update face positions for next frame
    face_positions = current_face_positions

    # ------------------------------
    # Display crowd analytics
    # ------------------------------
    # Percentage of each emotion across all faces
    all_emotions = [max(set(emotion_history[pid]), key=emotion_history[pid].count)
                    for pid in emotion_history if len(emotion_history[pid]) > 0]

    counts = {}
    total = len(all_emotions)
    for emo in all_emotions:
        counts[emo] = counts.get(emo, 0) + 1

    percentages = {k: round(v/total*100) for k,v in counts.items()} if total > 0 else {}

    cv2.putText(frame, f"People: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    y0 = 60
    for emo, pct in percentages.items():
        cv2.putText(frame, f"{emo}: {pct}%", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y0 += 30

    # Show frame
    cv2.imshow("Multi-Face Emotion Tracker", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
