import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Folder to store data
DATASET_PATH = "mudra_data"
os.makedirs(DATASET_PATH, exist_ok=True)

# Mudras to collect
mudras = ["chandrakala","pataka","mushti","shikara","mrigasirsha","alapadma","samyuta"]
data = {mudra: [] for mudra in mudras}

# Constants
TOTAL_SAMPLES_PER_MUDRA = 400
SAMPLES_PER_BREAK = 100
CAPTURE_DELAY = 0.08  # Faster image capture (80ms delay)
POSE_PREP_TIME = 5
BREAK_TIME_SMALL = 15  # 15s break after every 100 samples
BREAK_TIME_LARGE = 10  # 10s break after completing 400 samples

def extract_landmarks(results, width, height):
    """Extract hand landmarks and return bounding box coordinates."""
    hand_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [l.x * width for l in hand_landmarks.landmark]
        y_coords = [l.y * height for l in hand_landmarks.landmark]
        
        # Bounding box around the hand
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        hand_data.append({
            "landmarks": x_coords + y_coords,
            "bbox": (x_min, y_min, x_max, y_max)
        })
    return hand_data

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected hand."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def save_annotated_image(frame, mudra, count):
    """Save the frame with annotations."""
    folder_path = os.path.join(DATASET_PATH, mudra)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"{mudra}_{count}.jpg")
    cv2.imwrite(filename, frame)

def draw_timer(frame, seconds_left, text_prefix="", position=(10, 150)):
    """Draw countdown timer on frame."""
    color = (0, 255, 0) if seconds_left > 3 else (0, 0, 255)
    cv2.putText(frame, f"{text_prefix}{seconds_left}s", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cap = cv2.VideoCapture(0)
print("Press 'Q' to skip to next mudra | 'ESC' to quit")

for mudra in mudras:
    print(f"\nPreparing to collect samples for: {mudra}")
    prep_end_time = datetime.now() + timedelta(seconds=POSE_PREP_TIME)
    while datetime.now() < prep_end_time:
        ret, frame = cap.read()
        if not ret:
            break
        seconds_left = int((prep_end_time - datetime.now()).total_seconds())
        cv2.putText(frame, f"Prepare for: {mudra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        draw_timer(frame, seconds_left, "Starting in: ")
        cv2.imshow("Mudra Collection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"\nCollecting samples for: {mudra}")
    last_capture_time = time.time()
    count = 0
    
    while count < TOTAL_SAMPLES_PER_MUDRA:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_data in extract_landmarks(results, width, height):
                draw_bounding_box(frame, hand_data["bbox"])
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                if current_time - last_capture_time >= CAPTURE_DELAY:
                    data[mudra].append(hand_data["landmarks"])
                    save_annotated_image(frame, mudra, count)
                    count += 1
                    last_capture_time = current_time
        
        progress_bar = f"[{'=' * int(count/TOTAL_SAMPLES_PER_MUDRA * 20):{20}}]"
        cv2.putText(frame, f"Mudra: {mudra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Progress: {count}/{TOTAL_SAMPLES_PER_MUDRA}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, progress_bar, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Mudra Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        # 15-sec break after every 100 samples
        if count % SAMPLES_PER_BREAK == 0 and count != 0:
            print(f"\nTaking a {BREAK_TIME_SMALL}-second break...")
            break_end_time = datetime.now() + timedelta(seconds=BREAK_TIME_SMALL)
            while datetime.now() < break_end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                seconds_left = int((break_end_time - datetime.now()).total_seconds())
                cv2.putText(frame, "Break Time", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                draw_timer(frame, seconds_left, "Resuming in: ")
                cv2.imshow("Mudra Collection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    # 10-sec break after completing 400 images
    if count >= TOTAL_SAMPLES_PER_MUDRA:
        print(f"\nCompleted {TOTAL_SAMPLES_PER_MUDRA} samples for {mudra}. Taking a {BREAK_TIME_LARGE}-second break...")
        break_end_time = datetime.now() + timedelta(seconds=BREAK_TIME_LARGE)
        while datetime.now() < break_end_time:
            ret, frame = cap.read()
            if not ret:
                break
            seconds_left = int((break_end_time - datetime.now()).total_seconds())
            cv2.putText(frame, "Break Time", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            draw_timer(frame, seconds_left, "Next mudra in: ")
            cv2.imshow("Mudra Collection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

# Save data
with open(os.path.join(DATASET_PATH, "mudra_samples.json"), "w") as f:
    json.dump(data, f)
print("\nData collection complete!")

# Print summary
print("\nCollection Summary:")
for mudra, samples in data.items():
    print(f"{mudra}: {len(samples)} samples")
