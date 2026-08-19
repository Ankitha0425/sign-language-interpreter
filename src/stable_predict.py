import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

MODEL_PATH = "models/sign_language_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera")
    exit()

prediction_history = deque(maxlen=10)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        left = np.zeros(63)
        right = np.zeros(63)

        uses_two_hands = 0

        if len(results.multi_hand_landmarks) == 2:
            uses_two_hands = 1

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label

            values = []

            for landmark in hand_landmarks.landmark:
                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            if label == "Left":
                left = np.array(values)

            else:
                right = np.array(values)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        features = np.concatenate([
            [uses_two_hands],
            left,
            right
        ])

        features = features.reshape(1, -1)

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)[0]

        prediction_history.append(
            int(prediction)
        )

        most_common = Counter(
            prediction_history
        ).most_common(1)[0][0]

        letter = chr(
            ord("A") + most_common
        )

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled
            )

            confidence = np.max(
                probabilities
            ) * 100

        cv2.putText(
            frame,
            f"Prediction: {letter}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3
        )

        if confidence is not None:

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

    else:

        prediction_history.clear()

        cv2.putText(
            frame,
            "Show your hand",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Indian Sign Language Interpreter",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()