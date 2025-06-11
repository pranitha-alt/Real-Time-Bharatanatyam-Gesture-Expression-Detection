import cv2
import numpy as np
import mediapipe as mp
import json
from scipy.spatial.distance import cosine
from collections import deque


class MudraExpressionRecognizer:
    def __init__(self, video_path):
        # Load mudra data
        with open('mudra_data/mudra_samples.json', 'r') as f:
            self.mudra_samples = json.load(f)

        self.video_path = video_path
        self.mudra_buffer = deque(maxlen=10)
        self.expression_buffer = deque(maxlen=10)

        # Default values
        self.current_mudra = "a silent gesture of stillness"
        self.current_expression = "a peaceful stillness (Neutral)"

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.8)

        self.mp_drawing = mp.solutions.drawing_utils

    def extract_hand_landmarks(self, landmarks, width, height):
        return [lm.x * width for lm in landmarks.landmark] + [lm.y * height for lm in landmarks.landmark]

    def compare_with_samples(self, input_landmarks):
        best_match = None
        best_score = float('inf')
        for mudra_name, samples in self.mudra_samples.items():
            for sample in samples:
                if len(sample) != len(input_landmarks):
                    continue
                score = cosine(input_landmarks, sample)
                if score < best_score:
                    best_score = score
                    best_match = mudra_name
        return best_match if best_score < 0.15 else "a silent gesture of stillness"

    def get_stable_prediction(self, buffer):
        if not buffer:
            return None
        return max(set(buffer), key=buffer.count)

    def detect_expression(self, face_landmarks, width, height):
        ids = [13, 14, 159, 145, 67, 291]
        points = {i: (face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height) for i in ids}

        lip_distance = abs(points[13][1] - points[14][1])
        eye_openness = abs(points[159][1] - points[145][1])
        eye_horizontal = points[67][0] < points[291][0]

        if lip_distance > 15:
            return "wide-eyed wonder (Angry)"
        elif eye_openness < 3 and lip_distance < 3:
            return "a quiet heart (Feeling down)"
        elif lip_distance < 2 and eye_openness > 7:
            return "calm serenity (Neutral)"
        elif lip_distance < 7 and eye_horizontal:
            return "a soulful hush (Sad)"
        else:
            return "radiant joy (Happy)"

    def draw_bounding_box(self, frame, landmarks, width, height, color):
        x_vals = [int(lm.x * width) for lm in landmarks.landmark]
        y_vals = [int(lm.y * height) for lm in landmarks.landmark]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), color, 3)

    def draw_dashboard(self, frame, mudra, expression, color):
        panel_height = 120
        width, height = frame.shape[1], frame.shape[0]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        text_1 = f"You're channeling the grace of {mudra.lower()}"
        text_2 = f"with the emotion of {expression.lower()}."

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        text_size_1 = cv2.getTextSize(text_1, font, font_scale, thickness)[0]
        text_size_2 = cv2.getTextSize(text_2, font, font_scale, thickness)[0]

        x1 = (width - text_size_1[0]) // 2
        x2 = (width - text_size_2[0]) // 2

        cv2.putText(frame, text_1, (x1, 45), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text_1, (x1, 45), font, font_scale, color, thickness)

        cv2.putText(frame, text_2, (x2, 85), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text_2, (x2, 85), font, font_scale, color, thickness)

    def recognize_from_video(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        target_width = 480

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original_height, original_width = frame.shape[:2]
            target_height = int(target_width * (original_height / original_width))
            frame = cv2.resize(frame, (target_width, target_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_result = self.hands.process(frame_rgb)
            face_result = self.face_mesh.process(frame_rgb)

            box_color = (0, 0, 255)

            # Hand detection
            if hands_result.multi_hand_landmarks:
                for hand in hands_result.multi_hand_landmarks:
                    landmarks = self.extract_hand_landmarks(hand, target_width, target_height)
                    predicted_mudra = self.compare_with_samples(landmarks)

                    if predicted_mudra:
                        self.mudra_buffer.append(predicted_mudra)

                    self.draw_bounding_box(frame, hand, target_width, target_height, (0, 255, 0))
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            stable_mudra = self.get_stable_prediction(self.mudra_buffer)
            if stable_mudra:
                self.current_mudra = stable_mudra
                box_color = (0, 255, 0)

            # Face expression detection
            if face_result.multi_face_landmarks:
                for face in face_result.multi_face_landmarks:
                    predicted_expression = self.detect_expression(face, target_width, target_height)
                    self.expression_buffer.append(predicted_expression)

            stable_expression = self.get_stable_prediction(self.expression_buffer)
            if stable_expression:
                self.current_expression = stable_expression

            text_color = (0, 255, 0) if box_color == (0, 255, 0) else (0, 0, 255)
            self.draw_dashboard(frame, self.current_mudra, self.current_expression, text_color)

            cv2.namedWindow("Mudra & Expression Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mudra & Expression Recognition", target_width, target_height)
            cv2.imshow("Mudra & Expression Recognition", frame)

            if cv2.waitKey(70) in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.mp4"
    recognizer = MudraExpressionRecognizer(video_path)
    recognizer.recognize_from_video()
