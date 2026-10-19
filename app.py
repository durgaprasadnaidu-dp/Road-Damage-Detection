
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO11s Road Damage Detection",
    page_icon="🚧",
    layout="wide",
)

st.title("🚧 Road Damage Detection")
st.write(
    "Detect longitudinal cracks, transverse cracks, alligator "
    "cracking, and potholes using a trained YOLO11s model."
)

MODEL_PATH = Path(__file__).parent / "models" / "road_damage" / "best.pt"

CLASS_NAMES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole",
}


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the YOLO model: {error}")
    st.stop()

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.35,
    step=0.05,
)

image_size = st.sidebar.selectbox(
    "Image size",
    options=[640, 800, 1024],
    index=0,
)

st.sidebar.caption(
    "Higher thresholds show fewer detections. "
    "Confidence scores are not guarantees of correctness."
)

uploaded_file = st.file_uploader(
    "Upload a road image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
)

if uploaded_file is not None:
    is_image = uploaded_file.type.startswith("image")

    if is_image:
        image_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            dtype=np.uint8,
        )
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not read this image. Try another file.")
            st.stop()

        with st.spinner("Detecting road damage..."):
            result = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )[0]

        annotated = result.plot()

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with right:
            st.subheader("Detection Result")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                coords = [
                    round(float(value), 1)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "Class": CLASS_NAMES.get(
                            class_id, str(class_id)
                        ),
                        "Confidence": round(score, 4),
                        "X1": coords[0],
                        "Y1": coords[1],
                        "X2": coords[2],
                        "Y2": coords[3],
                    }
                )

        st.subheader("Detection Summary")
        st.write(f"Total detections: {len(detections)}")

        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.info(
                "No damage detected above the selected confidence "
                "threshold. This does not prove the road is damage-free."
            )

    else:
        st.subheader("Video Detection")

        if st.button("Run video detection"):
            temp_path = None
            cap = None

            try:
                suffix = Path(uploaded_file.name).suffix or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error("Could not open this video.")
                    st.stop()

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )
                max_frames = 300

                if total_frames > max_frames:
                    st.warning(
                        "For responsiveness, this demo processes at "
                        f"most {max_frames} frames."
                    )

                frames_to_process = min(
                    total_frames if total_frames > 0 else max_frames,
                    max_frames,
                )

                progress = st.progress(0.0)
                preview = st.empty()
                detections = []
                frames_processed = 0
                frames_with_detections = 0

                while frames_processed < max_frames:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    result = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )[0]

                    if result.boxes is not None and len(result.boxes):
                        frames_with_detections += 1

                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            detections.append(
                                {
                                    "Frame": frames_processed,
                                    "Class": CLASS_NAMES.get(
                                        class_id, str(class_id)
                                    ),
                                    "Confidence": round(
                                        float(box.conf.item()), 4
                                    ),
                                }
                            )

                    if frames_processed % 5 == 0:
                        preview.image(
                            cv2.cvtColor(
                                result.plot(), cv2.COLOR_BGR2RGB
                            ),
                            caption=f"Frame {frames_processed}",
                            use_container_width=True,
                        )

                    frames_processed += 1
                    progress.progress(
                        min(
                            frames_processed / frames_to_process,
                            1.0,
                        )
                    )

                st.success("Video processing finished.")
                st.write(f"Frames processed: {frames_processed}")
                st.write(
                    "Frames with detections: "
                    f"{frames_with_detections}"
                )

                if detections:
                    st.subheader("Video Detection Details")
                    st.dataframe(
                        detections, use_container_width=True
                    )
                else:
                    st.info(
                        "No damage detected above the selected "
                        "confidence threshold."
                    )

            except Exception as error:
                st.error(f"Video processing failed: {error}")

            finally:
                if cap is not None:
                    cap.release()

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

