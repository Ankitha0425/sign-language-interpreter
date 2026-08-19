import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av

from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Indian Sign Language Interpreter",
    page_icon="🤟",
    layout="wide"
)

MODEL_PATH = "models/sign_language_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.history = deque(maxlen=10)

        self.current_letter = ""

        self.confidence = 0.0

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            left = np.zeros(63)
            right = np.zeros(63)

            uses_two_hands = 0

            if len(
                results.multi_hand_landmarks
            ) == 2:

                uses_two_hands = 1

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                label = handedness.classification[
                    0
                ].label

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

                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
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

            scaled = scaler.transform(
                features
            )

            prediction = model.predict(
                scaled
            )[0]

            self.history.append(
                int(prediction)
            )

            stable_prediction = Counter(
                self.history
            ).most_common(1)[0][0]

            self.current_letter = chr(
                ord("A") + stable_prediction
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    scaled
                )

                self.confidence = float(
                    np.max(probabilities) * 100
                )

            cv2.putText(
                image,
                f"Prediction: {self.current_letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                image,
                f"Confidence: {self.confidence:.1f}%",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            self.history.clear()

            self.current_letter = ""

            self.confidence = 0.0

            cv2.putText(
                image,
                "Show your hand",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


st.markdown(
    """
    <h1 style="text-align:center;">
    🤟 Indian Sign Language Interpreter
    </h1>

    <p style="text-align:center;">
    Real-time sign recognition using Computer Vision,
    MediaPipe and Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

left_column, right_column = st.columns(
    [2, 1]
)

with left_column:

    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

with right_column:

    st.subheader("Prediction")

    prediction_box = st.empty()

    confidence_box = st.empty()

    st.divider()

    st.subheader("Recognized Text")

    if "text" not in st.session_state:

        st.session_state.text = ""

    text_box = st.empty()

    st.divider()

    st.info(
        """
        **How to use**

        1. Click START.
        2. Allow camera permission.
        3. Show your hand.
        4. The model predicts the gesture.
        """
    )

if ctx.video_processor:

    processor = ctx.video_processor

    if processor.current_letter:

        prediction_box.markdown(
            f"""
            <div style="
                text-align:center;
                padding:25px;
                border-radius:15px;
                background:#f0f2f6;
            ">

            <h4>Current Sign</h4>

            <h1 style="font-size:60px;">
            {processor.current_letter}
            </h1>

            </div>
            """,
            unsafe_allow_html=True
        )

        confidence_box.metric(
            "Confidence",
            f"{processor.confidence:.1f}%"
        )

    else:

        prediction_box.info(
            "Show your hand to the camera."
        )

    text_box.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background:#f0f2f6;
            font-size:25px;
            font-weight:bold;
        ">
        {st.session_state.text or "Waiting for signs..."}
        </div>
        """,
        unsafe_allow_html=True
    )