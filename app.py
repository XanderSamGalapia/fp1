import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO

@st.cache_resource(allow_output_mutation=True)
def load_models():
    pose_model = YOLO('weights/best.pt')
    gru_model = tf.keras.models.load_model('cheating_gru_model(0.88ac, 0.97 val).keras')
    return pose_model, gru_model

def extract_keypoints_from_frame(pose_model, frame, expected_keypoints=17):
    results = pose_model(frame)
    poses = results[0].keypoints.data.cpu().numpy()  # (num_people, 17, 3)
    if poses.shape[0] == 0:
        return np.zeros(expected_keypoints * 2)  # no person detected
    pose = poses[0]
    if pose.shape[0] != expected_keypoints:
        return np.zeros(expected_keypoints * 2)
    return pose[:, :2].flatten()

def process_video(pose_model, video_path, sequence_length=30, expected_keypoints=17):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = extract_keypoints_from_frame(pose_model, frame_rgb, expected_keypoints)
        keypoints_all.append(keypoints)
    cap.release()
    keypoints_array = np.array(keypoints_all)
    sequences = []
    for i in range(len(keypoints_array) - sequence_length + 1):
        seq = keypoints_array[i:i+sequence_length]
        if seq.shape == (sequence_length, expected_keypoints * 2):
            sequences.append(seq)
    return np.array(sequences)

# --- Streamlit UI ---
st.title("Cheating Detection with GRU on Pose Keypoints")

pose_model, gru_model = load_models()

uploaded_video = st.file_uploader("Upload a video for cheating detection", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.video(temp_video_path)

    with st.spinner("Extracting keypoints and predicting..."):
        sequences = process_video(pose_model, temp_video_path)
        if len(sequences) == 0:
            st.error("Not enough frames/keypoints detected for prediction.")
        else:
            preds = gru_model.predict(sequences)
            pred_classes = np.argmax(preds, axis=1)
            unique, counts = np.unique(pred_classes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            label_map = {0: "Normal", 1: "Cheating"}
            confidence = counts.max() / counts.sum()
            st.success(f"Prediction: **{label_map[majority_class]}** with confidence {confidence:.2f}")
