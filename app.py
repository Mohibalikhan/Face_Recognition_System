import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

st.title("📸 Face Recognition Attendance")

# Load known student faces
path = 'known_faces'
images = []
names = []

# Ensure the known_faces directory exists
if not os.path.exists(path):
    os.makedirs(path)

for filename in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{filename}")
    encoding = face_recognition.face_encodings(img)[0]
    images.append(encoding)
    names.append(os.path.splitext(filename)[0])  # Ali.jpg → Ali

# Create CSV if not exists
if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Time"]).to_csv("attendance.csv", index=False)

# Use session state to control camera
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

start = st.button("Start Camera")
stop = st.button("Close Camera")

if start:
    st.session_state.camera_running = True

if stop:
    st.session_state.camera_running = False

# New student registration section
st.subheader("Register New Student")
roll_number = st.text_input("Enter Roll Number (Without Slashes)")
uploaded_file = st.file_uploader("Upload Student Image", type=["jpg", "jpeg", "png"])

if st.button("Register Student"):
    if uploaded_file is not None and roll_number:
        # Check if the student is already registered
        if roll_number in names:
            st.error("❌ Student is already registered. Please use a different roll number.")
        else:
            # Load the uploaded image
            img = face_recognition.load_image_file(uploaded_file)
            encoding = face_recognition.face_encodings(img)

            if encoding:
                # Save the image to the known_faces directory
                file_path = os.path.join(path, f"{roll_number}.jpg")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Append the new encoding and name
                images.append(encoding[0])
                names.append(roll_number)
                st.success(f"✅ Student {roll_number} registered successfully!")
            else:
                st.error("❌ No face found in the uploaded image. Please try again.")
    else:
        st.error("❌ Please enter a roll number and upload an image.")

# Initialize webcam when camera_running is True
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    marked = []

    while st.session_state.camera_running:
        success, frame = cap.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_current = face_recognition.face_locations(rgb)
        encodings_current = face_recognition.face_encodings(rgb, faces_current)

        for face_encoding, face_loc in zip(encodings_current, faces_current):
            matches = face_recognition.compare_faces(images, face_encoding)
            face_dist = face_recognition.face_distance(images, face_encoding)
            best_match_index = np.argmin(face_dist)

            if matches[best_match_index]:
                name = names[best_match_index]

                if name not in marked:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.DataFrame([[name, now]], columns=["Name", "Time"])
                    df.to_csv("attendance.csv", mode='a', index=False, header=False)
                    st.success(f"✅ {name} marked present at {now}")
                    marked.append(name)

                # Draw box & label
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
