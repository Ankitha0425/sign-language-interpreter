import cv2
import mediapipe as mp
import numpy as np
import joblib

MODEL_PATH = "models/sign_language_model.pkl"
SCALER_PATH = "models/scaler.pkl"

print("Loading model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")

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

print("Camera started.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    left_landmarks = np.zeros(63)
    right_landmarks = np.zeros(63)

    uses_two_hands = 0

    if results.multi_hand_landmarks:

        if len(results.multi_hand_landmarks) == 2:
            uses_two_hands = 1

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label

            landmarks = []

            for landmark in hand_landmarks.landmark:

                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            if label == "Left":
                left_landmarks = np.array(
                    landmarks
                )

            elif label == "Right":
                right_landmarks = np.array(
                    landmarks
                )

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        features = np.concatenate([
            [uses_two_hands],
            left_landmarks,
            right_landmarks
        ])

        features = features.reshape(
            1,
            -1
        )

        features_scaled = scaler.transform(
            features
        )

        prediction = model.predict(
            features_scaled
        )

        predicted_class = int(
            prediction[0]
        )

        letter = chr(
            ord("A") + predicted_class
        )

        cv2.putText(
            frame,
            f"Prediction: {letter}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    else:

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

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Camera stopped.")