import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

class MudraExpressionRecognizer:
    def __init__(self):
        # Load trained mudra model and class labels
        self.model = tf.keras.models.load_model('mudra_data/mudra_model.h5')
        with open('mudra_data/mudra_classes.json', 'r') as f:
            self.mudra_classes = json.load(f)

        # Initialize MediaPipe hand and face solutions
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.8)

        self.mp_drawing = mp.solutions.drawing_utils

    def extract_hand_landmarks(self, hand_landmarks, width, height):
        return [l.x * width for l in hand_landmarks.landmark] + [l.y * height for l in hand_landmarks.landmark]

    def detect_expression(self, face_landmarks, width, height):
        key_points = {i: (face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height)
                      for i in [13, 14, 159, 145, 67, 291]}

        lip_distance = abs(key_points[13][1] - key_points[14][1])
        eye_openness = abs(key_points[159][1] - key_points[145][1])

        if lip_distance > 17:
            return "wide-eyed wonder (Surprise)"
        elif eye_openness < 4 and lip_distance < 3:
            return "a quiet heart (Feeling down)"
        elif lip_distance < 2 and eye_openness > 7:
            return "calm serenity (Neutral)"
        elif lip_distance < 7 and key_points[67][0] < key_points[291][0]:
            return "a soulful hush (Sad)"
        else:
            return "radiant joy (Happy)"

    def draw_bounding_box(self, frame, hand_landmarks, width, height, color):
        x_min, y_min = width, height
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            x_min, y_min = min(x, x_min), min(y, y_min)
            x_max, y_max = max(x, x_max), max(y, y_max)
        cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), color, 3)

    def draw_dashboard(self, frame, detected_mudra, detected_expression, text_color):
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - panel_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        display_text = f"You're channeling the grace of {detected_mudra.lower()} with the emotion of {detected_expression.lower()}."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        if text_size[0] > frame.shape[1] - 100:
            mudra_text = f"You're channeling the grace of {detected_mudra.lower()}"
            expression_text = f"with the emotion of {detected_expression.lower()}."

            cv2.putText(frame, mudra_text, (50, frame.shape[0] - 50), font, 0.6, (0, 0, 0), thickness + 2)
            cv2.putText(frame, mudra_text, (50, frame.shape[0] - 50), font, 0.6, text_color, thickness)

            cv2.putText(frame, expression_text, (50, frame.shape[0] - 25), font, 0.6, (0, 0, 0), thickness + 2)
            cv2.putText(frame, expression_text, (50, frame.shape[0] - 25), font, 0.6, text_color, thickness)
        else:
            cv2.putText(frame, display_text, (50, frame.shape[0] - 40), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, display_text, (50, frame.shape[0] - 40), font, font_scale, text_color, thickness)

    def recognize_live(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            results_hands = self.hands.process(image_rgb)
            results_face = self.face_mesh.process(image_rgb)

            detected_mudra = "a silent gesture of stillness"
            box_color = (0, 0, 255)

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    landmarks = self.extract_hand_landmarks(hand_landmarks, width, height)
                    landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    prediction = self.model.predict(landmarks, verbose=0)
                    confidence = np.max(prediction[0])
                    if confidence > 0.9:
                        detected_mudra = self.mudra_classes[np.argmax(prediction[0])]
                        box_color = (0, 255, 0)

                    self.draw_bounding_box(frame, hand_landmarks, width, height, box_color)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            detected_expression = "a peaceful stillness (Neutral)"
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    detected_expression = self.detect_expression(face_landmarks, width, height)

            text_color = (0, 255, 0) if box_color == (0, 255, 0) else (0, 0, 255)
            self.draw_dashboard(frame, detected_mudra, detected_expression, text_color)

            cv2.imshow("Mudra & Expression Recognition Dashboard", frame)
            if cv2.waitKey(10) in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = MudraExpressionRecognizer()
    recognizer.recognize_live()
