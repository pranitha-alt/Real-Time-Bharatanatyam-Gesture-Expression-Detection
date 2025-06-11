import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

class MudraRecognizer:
    def __init__(self):
        # Load the trained model and class labels
        self.model = tf.keras.models.load_model('mudra_data/mudra_model.h5')
        with open('mudra_data/mudra_classes.json', 'r') as f:
            self.mudra_classes = json.load(f)

        # Initialize Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, hand_landmarks, width, height):
        """Extract normalized hand landmark coordinates."""
        return [l.x * width for l in hand_landmarks.landmark] + \
               [l.y * height for l in hand_landmarks.landmark]

    def recognize_live(self):
        """Recognize mudras from live camera feed."""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks and predict
                    landmarks = self.extract_landmarks(hand_landmarks, width, height)
                    prediction = self.model.predict(np.array([landmarks]), verbose=0)
                    confidence = np.max(prediction[0])
                    class_index = np.argmax(prediction[0])

                    if confidence > 0.85:
                        predicted_class = self.mudra_classes[class_index]
                        color = (0, 255, 0)  # Green for recognized
                    else:
                        predicted_class = "Not Recognized"
                        color = (0, 0, 255)  # Red for unrecognized

                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Calculate bounding box
                    x_coords = [lm.x * width for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * height for lm in hand_landmarks.landmark]
                    x_min = int(max(0, min(x_coords) - 10))
                    x_max = int(min(width, max(x_coords) + 10))
                    y_min = int(max(0, min(y_coords) - 10))
                    y_max = int(min(height, max(y_coords) + 10))

                    # Draw bounding box with appropriate color
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Display prediction
                    cv2.putText(frame, f"Mudra: {predicted_class}", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Mudra Recognition", frame)
            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = MudraRecognizer()
    recognizer.recognize_live()
