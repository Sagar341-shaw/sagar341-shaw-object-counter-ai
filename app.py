import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", 
                "opencv-python-headless==4.8.0.74", "--quiet"], 
               capture_output=True)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import tempfile
import os
import time
import io
from PIL import Image

st.set_page_config(page_title="Object Counter AI", page_icon="🎯", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

COLORS = [
    (139, 92, 246), (99, 102, 241), (59, 130, 246), (16, 185, 129),
    (245, 158, 11), (239, 68, 68), (236, 72, 153), (14, 165, 233)
]

def get_color(label, label_colors, color_idx_ref):
    if label not in label_colors:
        label_colors[label] = COLORS[color_idx_ref[0] % len(COLORS)]
        color_idx_ref[0] += 1
    return label_colors[label]

def detect_and_count(image_np, model, conf, target_classes=None):
    results = model(image_np, conf=conf, verbose=False)[0]
    output = image_np.copy()
    counts = Counter()
    label_colors = {}
    color_idx_ref = [0]
    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        if target_classes and label not in target_classes:
            continue
        counts[label] += 1
        color = get_color(label, label_colors, color_idx_ref)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        tag = f'{label} #{counts[label]} {conf_score:.0%}'
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, y1-th-12), (x1+tw+10, y1), color, -1)
        cv2.putText(output, tag, (x1+5, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    H, W = output.shape[:2]
    total = sum(counts.values())
    cv2.rectangle(output, (0,0), (W,50), (10,10,20), -1)
    cv2.putText(output, f'OBJECTS DETECTED: {total}',
                (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (165,180,252), 2)
    return output, counts, label_colors

with st.sidebar:
    st.markdown("### 🎯 Object Counter AI")
    confidence = st.slider("Confidence", 0.1, 0.9, 0.45, 0.05)
    all_classes = ['person','car','bus','truck','motorcycle','bicycle',
                   'dog','cat','bird','bottle','chair','laptop',
                   'book','cup','bowl','umbrella','backpack']
    target_input = st.multiselect("Filter objects (empty = all)", options=all_classes)
    target_classes = target_input if target_input else None

st.markdown("## 🎯 Object Counting Detector")
st.markdown("AI-powered detection using YOLOv8 — IIT Mandi Internship")
st.markdown("---")

model = load_model()
st.success(f"✅ Model ready — detects {len(model.names)} object classes")

tab1, tab2 = st.tabs(["📁 Image Upload", "🎬 Video Upload"])

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png","webp"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.spinner("Detecting..."):
            start = time.time()
            result_img, counts, _ = detect_and_count(image, model, confidence, target_classes)
            elapsed = time.time() - start
        with col2:
            st.markdown("**Detected**")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("---")
        total = sum(counts.values())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Objects", total)
        c2.metric("Object Types", len(counts))
        c3.metric("Top Object", counts.most_common(1)[0][0] if counts else "—")
        c4.metric("Time", f"{elapsed:.2f}s")
        if counts:
            st.markdown("### Count Breakdown")
            for obj, cnt in counts.most_common():
                col_a, col_b, col_c = st.columns([2,5,1])
                col_a.markdown(f"🔹 **{obj}**")
                col_b.progress(cnt / max(counts.values()))
                col_c.markdown(f"**{cnt}**")
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("⬇️ Download Result", buf.getvalue(),
                           file_name="result.png", mime="image/png")

with tab2:
    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
    frame_skip = st.slider("Process every N frames", 1, 10, 3)
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        st.info(f"📹 Total frames: {total_frames}")
        if st.button("▶️ Start Processing"):
            cap = cv2.VideoCapture(tfile.name)
            frame_ph = st.empty()
            prog = st.progress(0)
            status = st.empty()
            all_counts = Counter()
            frame_idx = 0
            start = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    result_frame, counts, _ = detect_and_count(frame, model, confidence, target_classes)
                    all_counts.update(counts)
                    frame_ph.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                   use_container_width=True)
                frame_idx += 1
                prog.progress(min(frame_idx/max(total_frames,1), 1.0))
                status.markdown(f"Frame {frame_idx}/{total_frames} — Objects: {sum(all_counts.values())}")
            cap.release()
            os.unlink(tfile.name)
            st.success(f"✅ Done in {time.time()-start:.1f}s")
            st.markdown("### Final Count")
            for obj, cnt in all_counts.most_common():
                st.markdown(f"🔹 **{obj}**: {cnt}")
