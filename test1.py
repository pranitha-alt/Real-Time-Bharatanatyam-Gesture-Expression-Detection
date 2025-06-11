import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

class MudraRecognizer:
    def __init__(self, video_path):
        self.video_path = video_path  # Set video path

        # Load trained model and mudra class labels
        self.model = tf.keras.models.load_model('mudra_data/mudra_model.h5')
        with open('mudra_data/mudra_classes.json', 'r') as f:
            self.mudra_classes = json.load(f)

        # Initialize Mediapipe Hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, hand_landmarks, width, height):
        """Extracts hand landmark coordinates for model prediction."""
        return [l.x * width for l in hand_landmarks.landmark] + \
               [l.y * height for l in hand_landmarks.landmark]

    def draw_rounded_box(self, img, top_left, bottom_right, color, thickness=2, radius=10, transparency=0.3):
        """Draws a rounded rectangle with a semi-transparent background."""
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)  # Filled rectangle
        img = cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0)  # Apply transparency
        cv2.rectangle(img, top_left, bottom_right, color, thickness)  # Border
        return img

    def recognize_from_video(self):
        """Detect and recognize mudras from a video file without resizing the video."""
        cap = cv2.VideoCapture(self.video_path)

        # Get original video dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit when video ends

            # Convert frame to RGB for Mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get bounding box coordinates
                    x_min = max(0, min([lm.x for lm in hand_landmarks.landmark]) * original_width - 20)
                    x_max = min(original_width, max([lm.x for lm in hand_landmarks.landmark]) * original_width + 20)
                    y_min = max(0, min([lm.y for lm in hand_landmarks.landmark]) * original_height - 20)
                    y_max = min(original_height, max([lm.y for lm in hand_landmarks.landmark]) * original_height + 20)

                    # Extract landmarks for prediction
                    landmarks = self.extract_landmarks(hand_landmarks, original_width, original_height)
                    prediction = self.model.predict(np.array([landmarks]), verbose=0)
                    confidence = np.max(prediction[0])

                    # Determine recognized mudra
                    if confidence > 0.85:
                        predicted_class = self.mudra_classes[np.argmax(prediction[0])]
                        box_color = (0, 255, 0)  # Green for recognized
                        text_color = (0, 0, 0)
                    else:
                        predicted_class = "Not Recognized"
                        box_color = (0, 0, 255)  # Red for unrecognized
                        text_color = (255, 255, 255)

                    # Draw bounding box
                    frame = self.draw_rounded_box(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, thickness=3)

                    # Display label with background
                    label_size = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    label_x = int(x_min)
                    label_y = int(y_min) - 10
                    label_w, label_h = label_size

                    cv2.rectangle(frame, (label_x, label_y - label_h - 5), (label_x + label_w + 10, label_y + 5), box_color, -1)
                    cv2.putText(frame, predicted_class, (label_x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            # Display the frame in its original size
            cv2.namedWindow("Mudra Recognition from Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mudra Recognition from Video", original_width, original_height)
            cv2.imshow("Mudra Recognition from Video", frame)

            key = cv2.waitKey(10)  # Adjust frame rate for better viewing
            if key == ord('q') or key == 27:  # Press 'q' or 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video6.mp4"  # Replace with your video file path
    recognizer = MudraRecognizer(video_path)
    recognizer.recognize_from_video()