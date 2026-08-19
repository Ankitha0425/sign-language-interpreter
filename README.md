# 🤟 Indian Sign Language Interpreter

A real-time Indian Sign Language (ISL) gesture recognition system that uses Computer Vision and Machine Learning to recognize hand gestures through a webcam and convert them into understandable output.

## 📸 Demo

Real-time Indian Sign Language gesture recognition using webcam and MediaPipe.

![Indian Sign Language Interpreter Demo](screenshots/demo.png)

## 🚀 Features

- Real-time hand gesture recognition
- Webcam-based interaction
- Indian Sign Language gesture classification
- MediaPipe hand landmark detection
- Machine Learning-based gesture prediction
- Support for 26 gesture classes
- Text-to-speech output
- Streamlit web interface
- Stable prediction using consecutive frames

## 📊 Model Performance

Two machine learning classifiers were evaluated:

| Model | Validation Accuracy |
|---|---:|
| Random Forest | 99.84% |
| SVM | **99.87%** |

The SVM model achieved the best validation performance and was selected as the final classifier.

### Final Test Performance

- **Test Accuracy:** 99.88%
- **Test Samples:** 7,629
- **Number of Classes:** 26
- **Training Samples:** 35,601
- **Validation Samples:** 7,629
- **Input Features:** 127

The classification results show consistently high precision, recall, and F1-score across the 26 gesture classes.

## 🛠️ Technologies Used

- **Python 3.12**
- **OpenCV** – real-time webcam capture
- **MediaPipe** – hand landmark detection
- **NumPy & Pandas** – data processing
- **Scikit-learn** – machine learning
- **SVM** – final gesture classification model
- **Streamlit** – web application
- **pyttsx3** – text-to-speech

## 📁 Project Structure

```text
sign-language-interpreter/
│
├── data/
│   └── sign_landmarks.csv
│
├── models/
│   ├── hand_landmarker.task
│   ├── scaler.pkl
│   └── sign_language_model.pkl
│
├── screenshots/
│   └── demo.png
│
├── src/
│   ├── app.py
│   ├── collect_data.py
│   ├── hand_test.py
│   ├── inspect_dataset.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── stable_predict.py
│   └── train.py
│
├── requirements.txt
├── streamlit_app.py
└── README.md

## 🌐 Live Demo

Try the deployed Indian Sign Language Interpreter:

🚀 [Launch the Application](https://sign-language-interpreter-xyw9cdfu8hewdglcfocaym.streamlit.app/)

## 🌐 Live Demo

🚀 **[Launch Indian Sign Language Interpreter](https://sign-language-interpreter-xyw9cdfu8hewdglcfocaym.streamlit.app/)**

The application provides real-time Indian Sign Language gesture recognition through a webcam using MediaPipe and an SVM classifier.