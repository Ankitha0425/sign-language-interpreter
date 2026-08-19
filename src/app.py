import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
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

engine = pyttsx3.init()

prediction_history = deque(maxlen=10)

recognized_text = ""

last_letter = ""
last_added_time = 0

ADD_DELAY = 1.5

print("======================================")
print("INDIAN SIGN LANGUAGE INTERPRETER")
print("======================================")
print()
print("Controls:")
print("SPACE  - Add space")
print("B      - Backspace")
print("C      - Clear text")
print("S      - Speak")
print("Q      - Quit")
print()

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    current_letter = None
    confidence = 0

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

        features = features.reshape(
            1,
            -1
        )

        scaled_features = scaler.transform(
            features
        )

        prediction = model.predict(
            scaled_features
        )[0]

        prediction_history.append(
            int(prediction)
        )

        stable_prediction = Counter(
            prediction_history
        ).most_common(1)[0][0]

        current_letter = chr(
            ord("A") + stable_prediction
        )

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_features
            )

            confidence = (
                np.max(probabilities) * 100
            )

        current_time = time.time()

        if (
            current_letter != last_letter
            and current_time - last_added_time > ADD_DELAY
            and confidence >= 70
        ):

            recognized_text += current_letter

            last_letter = current_letter
            last_added_time = current_time

    else:

        prediction_history.clear()

    cv2.rectangle(
        frame,
        (10, 10),
        (700, 160),
        (0, 0, 0),
        -1
    )

    if current_letter:

        cv2.putText(
            frame,
            f"Prediction: {current_letter}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"Text: {recognized_text}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE: space | B: delete | C: clear | S: speak | Q: quit",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Indian Sign Language Interpreter",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        recognized_text += " "

        last_letter = ""

    elif key == ord("b"):

        recognized_text = recognized_text[:-1]

    elif key == ord("c"):

        recognized_text = ""
        last_letter = ""

    elif key == ord("s"):

        if recognized_text.strip():

            print(
                "Speaking:",
                recognized_text
            )

            engine.say(
                recognized_text
            )

            engine.runAndWait()

    elif key == ord("q"):

        break

cap.release()

cv2.destroyAllWindows()

hands.close()

engine.stop()