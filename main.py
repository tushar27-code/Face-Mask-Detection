
from history import save_detection, read_history
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading
import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

# -------------------------------
# PATH CONFIGURATION
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mask.keras"
CASCADE_PATH = BASE_DIR / "face.xml"
SAVE_DIR = BASE_DIR / "Fetched_img"

SAVE_DIR.mkdir(exist_ok=True)

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# -------------------------------
# LOAD MODEL AND FACE DETECTOR
# -------------------------------

@st.cache_resource
def load_detection_model():
    model = load_model(MODEL_PATH)
    face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if face_detector.empty():
        raise RuntimeError("Could not load face.xml")

    return model, face_detector


try:
    model, face_detector = load_detection_model()
except Exception as e:
    st.error(f"Error loading detection files: {e}")
    st.stop()

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("😷 Face Mask Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "Live Webcam",
        "IP Webcam"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("AI-based Face Mask Detection System")

# -------------------------------
# HOME PAGE
# -------------------------------

if page == "Home":

    st.title("😷 Face Mask Detection System")

    st.write(
        "Welcome to the Face Mask Detection application."
    )

    st.markdown("""
    ### Available features

    - Image-based face mask detection
    - Identification of people wearing masks
    - Identification of people not wearing masks
    - Save detected no-mask face images
    """)

    st.success("Detection model loaded successfully!")

# -------------------------------
# IMAGE DETECTION
# -------------------------------

elif page == "Image Detection":

    st.title("📷 Image Face Mask Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("Unable to read this image.")
            st.stop()

        result_image = image.copy()

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        mask_count = 0
        no_mask_count = 0

        for (x, y, w, h) in faces:

            face = image[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (150, 150)
            )

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(
                face,
                verbose=0
            )[0][0]

            # Class mapping:
            # with_mask = 0
            # without_mask = 1

            if prediction >= 0.5:

                label = "NO MASK"
                color = (0, 0, 255)
                no_mask_count += 1

                filename = (
                    f"no_mask_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )

                save_path = SAVE_DIR / filename

                cv2.imwrite(
                    str(save_path),
                    image[y:y+h, x:x+w]
                )

            else:

                label = "MASK"
                color = (0, 255, 0)
                mask_count += 1

            cv2.rectangle(
                result_image,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                result_image,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )


        # Save this image detection to history
        save_detection("Image", mask_count, no_mask_count)

        # -------------------------------
        # DISPLAY RESULTS
        # -------------------------------

   
        st.subheader("Detection Result")

        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Faces", len(faces))
        col2.metric("With Mask", mask_count)
        col3.metric("Without Mask", no_mask_count)

        if len(faces) == 0:
            st.warning("No faces detected in this image.")

        elif no_mask_count > 0:
            st.error(f"{no_mask_count} face(s) detected without a mask.")

        else:
            st.success("Everyone detected is wearing a mask.")

        st.caption(
            "No-mask face crops are saved in the Fetched_img folder. "
            "Use images only where you have appropriate permission."
        )


# -------------------------------
# VIDEO DETECTION
# -------------------------------

elif page == "Video Detection":

    st.title("🎥 Video Face Mask Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Detection"):

            import tempfile

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_file.write(uploaded_video.getbuffer())
            temp_file.close()

            video = cv2.VideoCapture(temp_file.name)

            if not video.isOpened():

                st.error("Unable to open video.")

            else:

                stframe = st.empty()
                progress = st.progress(0)
                status = st.empty()

                total_frames = int(
                    video.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if total_frames <= 0:
                    st.error("Video contains no readable frames.")

                else:

                    total_mask = 0
                    total_no_mask = 0
                    frame_number = 0

                    while True:

                        ret, frame = video.read()

                        if not ret:
                            break

                        frame_number += 1

                        gray = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2GRAY
                        )

                        faces = face_detector.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=4,
                            minSize=(30, 30)
                        )

                        for (x, y, w, h) in faces:

                            face_crop = frame[y:y+h, x:x+w]

                            face_crop = cv2.resize(
                                face_crop,
                                (150, 150)
                            )

                            face_crop = cv2.cvtColor(
                                face_crop,
                                cv2.COLOR_BGR2RGB
                            )

                            face_crop = (
                                face_crop.astype("float32") / 255.0
                            )

                            face_crop = np.expand_dims(
                                face_crop,
                                axis=0
                            )

                            prediction = model.predict(
                                face_crop,
                                verbose=0
                            )[0][0]

                            if prediction >= 0.5:

                                label = "NO MASK"
                                color = (0, 0, 255)
                                total_no_mask += 1

                            else:

                                label = "MASK"
                                color = (0, 255, 0)
                                total_mask += 1

                            cv2.rectangle(
                                frame,
                                (x, y),
                                (x+w, y+h),
                                color,
                                3
                            )

                            cv2.putText(
                                frame,
                                label,
                                (x, max(y-10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2
                            )

                        # Display annotated video frame
                        stframe.image(
                            frame,
                            channels="BGR",
                            width="stretch"
                        )

                        progress.progress(
                            min(frame_number / total_frames, 1.0)
                        )

                        status.write(
                            f"Processing frame: {frame_number}/{total_frames}"
                        )

                    video.release()

                    st.success("Video detection completed!")

                    st.subheader("Detection Report")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Frames",
                        frame_number
                    )

                    col2.metric(
                        "Mask Detections",
                        total_mask
                    )

                    col3.metric(
                        "No-Mask Detections",
                        total_no_mask
                    )

                    st.info(
                        "Counts represent face detections across all frames, "
                        "not unique individuals."
                    )

            video.release()

            os.remove(temp_file.name)

# -------------------------------
# LIVE WEBCAM DETECTION
# -------------------------------

elif page == "Live Webcam":

    st.title("📹 Live Webcam Face Mask Detection")

    st.write(
        "Allow camera access in your browser to start live detection."
    )

    class FaceMaskProcessor(VideoProcessorBase):

        def __init__(self):
            self.mask_count = 0
            self.no_mask_count = 0
            self.lock = threading.Lock()

        def recv(self, frame):

            image = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            current_mask = 0
            current_no_mask = 0

            for (x, y, w, h) in faces:

                face = image[y:y+h, x:x+w]

                face = cv2.resize(face, (150, 150))

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)

                prediction = model.predict(
                    face,
                    verbose=0
                )[0][0]

                if prediction >= 0.5:

                    label = "NO MASK"
                    color = (0, 0, 255)
                    current_no_mask += 1

                else:

                    label = "MASK"
                    color = (0, 255, 0)
                    current_mask += 1

                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                cv2.putText(
                    image,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            # Accumulate session detections
            with self.lock:
                self.mask_count += current_mask
                self.no_mask_count += current_no_mask

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

    ctx = webrtc_streamer(
        key="face-mask-webcam",
        video_processor_factory=FaceMaskProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    # Display session statistics
    st.markdown("---")
    st.subheader("📊 Live Session Statistics")

    if ctx.video_processor:

        with ctx.video_processor.lock:

            total_mask = ctx.video_processor.mask_count
            total_no_mask = ctx.video_processor.no_mask_count

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "With Mask",
            total_mask
        )

        col2.metric(
            "Without Mask",
            total_no_mask
        )

        col3.metric(
            "Total Detections",
            total_mask + total_no_mask
        )

    else:
        st.info("Start your webcam to view session statistics.")

        
    # Save webcam session report
    if ctx.video_processor:

        if st.button("💾 Save Webcam Session Report"):

            with ctx.video_processor.lock:

                session_mask = ctx.video_processor.mask_count
                session_no_mask = ctx.video_processor.no_mask_count

            save_detection(
                "Live Webcam",
                session_mask,
                session_no_mask
            )

            st.success("Webcam session report saved successfully!")


    st.markdown("---")

    st.subheader("Webcam Instructions")

    st.write("""
    1. Click START to activate your webcam.
    2. Allow camera permission in your browser.
    3. Show your face to the camera.
    4. View cumulative detection statistics.
    5. Click STOP to turn off your webcam.
    """)

    st.info(
        "Statistics count face detections across frames, "
        "not unique individuals. Session history saving "
        "will be added in the next step."
    )
# -------------------------------
# IP WEBCAM DETECTION
# -------------------------------

elif page == "IP Webcam":

    st.title("📱 Android IP Webcam Detection")

    st.write(
        "Connect your Android phone camera using the IP Webcam app."
    )

    ip_url = st.text_input(
        "Enter IP Webcam video URL",
        placeholder="http://192.168.1.105:8080/video"
    )

    if st.button("Connect IP Webcam"):

        if not ip_url.strip():
            st.error("Please enter your IP Webcam URL.")

        else:

            cap = cv2.VideoCapture(ip_url.strip())

            if not cap.isOpened():
                st.error(
                    "Unable to connect. Check the URL and Wi-Fi connection."
                )

                cap.release()

            else:

                st.session_state["ip_camera_connected"] = True
                st.session_state["ip_camera_url"] = ip_url.strip()
                cap.release()

                st.success("IP Webcam connection successful!")

    if st.session_state.get("ip_camera_connected", False):

        st.markdown("---")

        st.success("Phone camera is connected.")

        run_detection = st.checkbox(
            "Start IP Webcam Detection",
            value=False
        )

        if run_detection:

            cap = cv2.VideoCapture(
                st.session_state["ip_camera_url"]
            )

            if not cap.isOpened():
                st.error("Camera connection lost.")

            else:

                frame_placeholder = st.empty()
                count_placeholder = st.empty()

                stop_button = st.button("Stop IP Webcam")

                while cap.isOpened():

                    if stop_button:
                        break

                    ret, frame = cap.read()

                    if not ret:
                        st.warning("Unable to receive video frames.")
                        break

                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    faces = face_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    mask_count = 0
                    no_mask_count = 0

                    for (x, y, w, h) in faces:

                        face = frame[y:y+h, x:x+w]

                        face = cv2.resize(
                            face,
                            (150, 150)
                        )

                        face = cv2.cvtColor(
                            face,
                            cv2.COLOR_BGR2RGB
                        )

                        face = face.astype("float32") / 255.0
                        face = np.expand_dims(face, axis=0)

                        prediction = model.predict(
                            face,
                            verbose=0
                        )[0][0]

                        if prediction >= 0.5:

                            label = "NO MASK"
                            color = (0, 0, 255)
                            no_mask_count += 1

                            # Save no-mask face crop
                            filename = (
                                f"ip_no_mask_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            )

                            cv2.imwrite(
                                str(SAVE_DIR / filename),
                                frame[y:y+h, x:x+w]
                            )

                        else:

                            label = "MASK"
                            color = (0, 255, 0)
                            mask_count += 1

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x+w, y+h),
                            color,
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2
                        )

                    frame_placeholder.image(
                        frame,
                        channels="BGR",
                        width="stretch"
                    )

                    count_placeholder.write(
                        f"With Mask: {mask_count} | Without Mask: {no_mask_count}"
                    )

                cap.release()

                st.info("IP Webcam detection stopped.")


# -------------------------------
# DASHBOARD
# -------------------------------


elif page == "Dashboard":

    st.title("📊 Face Mask Detection Dashboard")
    st.write("Detection history and application statistics.")

    import pandas as pd

    history = read_history()

    if len(history) == 0:
        st.info("No detection history available yet.")
    else:

        # Calculate totals
        total_mask = int(history["Mask Count"].sum())
        total_no_mask = int(history["No Mask Count"].sum())
        total_detections = total_mask + total_no_mask

        # Dashboard metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Detections", total_detections)
        col2.metric("With Mask", total_mask)
        col3.metric("Without Mask", total_no_mask)

        st.markdown("---")

        # Detection mode breakdown
        st.subheader("Detection Mode Breakdown")

        mode_data = history.groupby("Detection Mode")[
            ["Mask Count", "No Mask Count"]
        ].sum()

        st.bar_chart(mode_data)

        st.markdown("---")

        # Daily detection trend
        st.subheader("Daily Detection Trend")

        history["Date"] = pd.to_datetime(history["Date"])

        daily_data = history.groupby("Date")[
            ["Mask Count", "No Mask Count"]
        ].sum()

        st.line_chart(daily_data)

        st.markdown("---")

        # History table
        st.subheader("Detection History")

        st.dataframe(
            history.sort_values(
                by=["Date", "Time"],
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )

        # Download history
        st.subheader("Download Detection Report")

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Detection History CSV",
            data=csv_data,
            file_name="face_mask_detection_history.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # Saved image gallery
    st.subheader("📁 Saved No-Mask Face Images")

    saved_images = sorted(
        SAVE_DIR.glob("*.jpg"),
        reverse=True
    )

    st.metric("Saved Images", len(saved_images))

    if saved_images:

        cols = st.columns(3)

        for index, image_path in enumerate(saved_images):

            with cols[index % 3]:

                st.image(
                    str(image_path),
                    caption=image_path.name,
                    width="stretch"
                )

    else:
        st.info("No saved face images available.")


