import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import dlib
import threading
import time
from scipy.spatial import distance as dist
from imutils import face_utils
from pydub import AudioSegment
from pydub.playback import play
import os
# Load the trained CNN model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "my_model.keras")
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Ensure 'my_model.keras' is in the project folder.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()

alarm_sound = "alarm.wav"
if not os.path.exists(alarm_sound):
    st.error("❌'alarm.wav' not found! Add an alarm sound file to the project folder.")
    st.stop()

# Load dlib’s face detector & facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    st.error("❌ 'shape_predictor_68_face_landmarks.dat' not found! Download and place it in the project folder.")
    st.stop()
predictor = dlib.shape_predictor(predictor_path)

# EAR threshold and consecutive frame count
EYE_AR_THRESH = 0.2  # Threshold for eyes closed
BLINK_CONSEC_FRAMES = 3  # Ignore normal blinks
DROWSY_CONSEC_TIME = 5  # Trigger alarm only if drowsy for 5 seconds

# Function to compute EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)
# Function to preprocess the face for CNN model
def preprocess_frame(frame, face):
    x, y, w, h = face
    face_crop = frame[y:y+h, x:x+w]

    # ✅ Check if face_crop is empty before resizing
    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None  # Return None if no face detected

    face_crop = cv2.resize(face_crop, (224, 224))
    face_crop = np.expand_dims(face_crop, axis=0) / 255.0
    return face_crop

# Function to play alarm in a separate thread
def play_alarm():
    sound = AudioSegment.from_file(alarm_sound, format="wav")
    play(sound)

# Streamlit UI
st.title("🚗 Real-Time Driver Drowsiness Detection")
st.write("Click **Start Detection** to analyze in real-time.")

# Session state variables
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "alarm_triggered" not in st.session_state:
    st.session_state.alarm_triggered = False
if "drowsy_start_time" not in st.session_state:
    st.session_state.drowsy_start_time = None
if "blink_counter" not in st.session_state:
    st.session_state.blink_counter = 0

# Start Detection
if st.button("Start Detection"):
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
    st.session_state.camera_on = True
    st.session_state.drowsy_start_time = None
    st.session_state.alarm_triggered = False
    st.session_state.blink_counter = 0

# Stop Detection
if st.button("Stop Detection"):
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.camera_on = False
    st.session_state.drowsy_start_time = None
    st.session_state.alarm_triggered = False
    st.session_state.blink_counter = 0
    cv2.destroyAllWindows()
    st.success("✅ Camera Stopped Successfully!")

FRAME_WINDOW = st.empty()
ALERT_BOX = st.empty()
SCORE_BOX = st.empty()

# Real-time video loop
while st.session_state.camera_on:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Failed to capture image. Please check your webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye landmarks
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0  # Average of both eyes

        # Predict drowsiness using CNN model
        face_box = (face.left(), face.top(), face.width(), face.height())
        face_img = preprocess_frame(frame_rgb, face_box)
        cnn_score = model.predict(face_img, verbose=0)[0][0] if face_img is not None else 0

        # Combine EAR and CNN model prediction
        final_drowsiness_score = (ear + cnn_score) / 2.0

        # Display EAR and CNN prediction score
        SCORE_BOX.info(f"EAR: {ear:.2f} | CNN Score: {cnn_score:.2f} | Drowsiness: {final_drowsiness_score:.2f}")

        # Handle blinking: Ignore if eyes are closed for a short duration
        if ear < EYE_AR_THRESH:
            st.session_state.blink_counter += 1  # Count frames with closed eyes

            if st.session_state.blink_counter > BLINK_CONSEC_FRAMES:
                # If continuously closed for 5 seconds, trigger drowsiness
                if st.session_state.drowsy_start_time is None:
                    st.session_state.drowsy_start_time = time.time()

                elapsed_time = time.time() - st.session_state.drowsy_start_time

                if elapsed_time >= DROWSY_CONSEC_TIME and not st.session_state.alarm_triggered:
                    ALERT_BOX.error("🚨 Drowsiness Detected for 5 Seconds! WAKE UP! 🚨")
                    st.session_state.alarm_triggered = True
                    threading.Thread(target=play_alarm, daemon=True).start()
        else:
            st.session_state.blink_counter = 0  # Reset blink counter if eyes open
            st.session_state.drowsy_start_time = None
            st.session_state.alarm_triggered = False
            ALERT_BOX.empty()

    FRAME_WINDOW.image(frame_rgb, channels="RGB")
    time.sleep(0.01)  # Ultra-low delay for real-time responsiveness

# Ensure camera releases properly when stopped
if st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
    cv2.destroyAllWindows()
