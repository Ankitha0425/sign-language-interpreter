import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "data/sign_landmarks.csv"

os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

label = input("Enter sign label (A/B/C/L/Y): ").upper()

print()
print("Collecting data for:", label)
print("Press SPACE to save a sample.")
print("Press Q to quit.")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for landmark in hand_landmarks.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Save | Q = Quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Dataset Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            with open(
                DATA_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [label] + landmarks
                )

            print(f"Saved {label} sample")

        elif key == ord("q"):
            break

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Dataset Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Dataset collection completed.")