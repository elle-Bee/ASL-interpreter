import pickle
import cv2
import mediapipe as mp
import numpy as np
from labels_dict import labels_dict


class GestureClassifier:
    def __init__(self):
        self.model_dict = pickle.load(open("./model.p", "rb"))
        self.model = self.model_dict["model"]

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=True, min_detection_confidence=0.3
        )

    def predict(self, frame):
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)
        predicted_character = None  # Initialize to None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = self.model.predict(
                [np.asarray(data_aux + [0] * (84 - len(data_aux)))]
            )
            predicted_character = labels_dict[prediction[0]]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        return predicted_character, frame
