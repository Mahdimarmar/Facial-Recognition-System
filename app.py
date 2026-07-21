import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os
import plotly.graph_objects as go


# Page config

st.set_page_config(
    page_title="Age, Gender & Race Prediction",
    page_icon="🧠",
    layout="wide"
)


# Load model and face detector

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_model.keras", compile=False)

model = load_model()
race_labels = ["White", "Black", "Asian", "Indian", "Others"]

PROTOTXT_PATH   = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_URL    = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

@st.cache_resource
def load_face_detector():
    if not os.path.exists(PROTOTXT_PATH):
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(CAFFEMODEL_PATH):
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)
    return cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

net = load_face_detector()


# Sidebar navigation

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🔍 Predict", "📊 Model Results"])

# ==============================================================
# Helper — detect faces
# ==============================================================
def detect_faces(img_rgb, confidence_threshold=0.5):
    h, w = img_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_rgb, (300, 300)),
        scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2-x1, y2-y1, float(confidence)))
    return faces


# Helper — process image

def process_image(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces   = detect_faces(img_rgb)

    if len(faces) == 0:
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        st.error("No face detected! Please try a clearer image with a visible face.")
        return

    img_display = img_rgb.copy()
    for (x, y, w, h, conf) in faces:
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_display, f"{conf:.0%}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(img_display,
             caption=f"{len(faces)} face(s) detected",
             use_container_width=True)

    for i, (x, y, w, h, conf) in enumerate(faces):
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        y2 = min(img_bgr.shape[0], y + h + pad_y)

        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_resized    = cv2.resize(face_crop, (64, 64))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input      = np.expand_dims(face_normalized, axis=0)

        gender_pred, race_pred, age_pred = model.predict(face_input, verbose=0)
        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
        race   = race_labels[np.argmax(race_pred[0])]
        age    = max(1, int(round(age_pred[0][0])))

        st.subheader("Prediction Results" + (f" — Face {i+1}" if len(faces) > 1 else ""))
        st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), width=150, caption="Cropped Face")

        col1, col2, col3 = st.columns(3)
        col1.metric("Gender", gender)
        col2.metric("Race",   race)
        col3.metric("Age",    f"{age} yrs")

        with st.expander("Show confidence scores"):
            st.write(f"**Detection confidence:** {conf:.2%}")
            st.write(f"**Gender confidence:** {gender_pred[0][0]:.2%}")
            st.write("**Race probabilities:**")
            for label, prob in zip(race_labels, race_pred[0]):
                st.progress(float(prob), text=f"{label}: {prob:.2%}")

        st.divider()


# PAGE 1 — Predict

if page == "🔍 Predict":
    st.title("Age, Gender & Race Prediction")
    st.write("Detect faces and predict age, gender, and race using AI.")

    mode = st.radio(
        "Choose input method:",
        ["📁 Upload Image", "📷 Use Camera"],
        horizontal=True
    )
    st.divider()

    if mode == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
        if uploaded_file is not None:
            image   = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(image)
            process_image(img_rgb)

    elif mode == "📷 Use Camera":
        st.write("Take a photo and the app will detect and predict automatically.")
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image   = Image.open(camera_photo).convert("RGB")
            img_rgb = np.array(image)
            process_image(img_rgb)


# PAGE 2 — Model Results

elif page == "📊 Model Results":
    st.title("Model Training Results")
    st.write("Comparison between the Machine Learning (Random Forest) and Deep Learning (CNN) models.")
    st.divider()

    # ── Dataset Info ──────────────────────────────────────────
    st.subheader("Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Images", "23,705")
    d2.metric("Training Set", "18,964  (80%)")
    d3.metric("Test Set",     "4,741  (20%)")
    st.divider()

    # ── Graph 1: Gender & Race Accuracy ──────────────────────
    st.subheader("Gender & Race Accuracy — ML vs DL")

    fig1 = go.Figure()
    metrics  = ["Gender Accuracy", "Race Accuracy"]
    ml_vals  = [81.21, 63.34]
    dl_vals  = [91.77, 76.14]

    fig1.add_trace(go.Bar(
        name="ML — Random Forest",
        x=metrics,
        y=ml_vals,
        marker_color="#378ADD",
        text=[f"{v}%" for v in ml_vals],
        textposition="outside"
    ))
    fig1.add_trace(go.Bar(
        name="DL — CNN",
        x=metrics,
        y=dl_vals,
        marker_color="#1D9E75",
        text=[f"{v}%" for v in dl_vals],
        textposition="outside"
    ))
    fig1.update_layout(
        barmode="group",
        yaxis=dict(title="Accuracy (%)", range=[0, 110]),
        xaxis=dict(title="Metric"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444")
    )
    fig1.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── Graph 2: Age MAE ──────────────────────────────────────
    st.subheader("Age MAE — ML vs DL (lower is better)")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="ML — Random Forest",
        x=["Age MAE"],
        y=[12.15],
        marker_color="#378ADD",
        text=["12.15 yrs"],
        textposition="outside",
        width=0.3
    ))
    fig2.add_trace(go.Bar(
        name="DL — CNN",
        x=["Age MAE"],
        y=[10.20],
        marker_color="#1D9E75",
        text=["10.20 yrs"],
        textposition="outside",
        width=0.3
    ))
    fig2.update_layout(
        barmode="group",
        yaxis=dict(title="MAE (years)", range=[0, 16]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444")
    )
    fig2.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Graph 3: Race Dataset Distribution ───────────────────
    st.subheader("Dataset Race Distribution")
    st.write("The dataset is heavily imbalanced — this is why race prediction is harder than gender.")

    fig3 = go.Figure()
    races       = ["White", "Black", "Asian", "Indian", "Others"]
    percentages = [59, 18, 15, 5, 3]
    colors      = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#888780"]

    fig3.add_trace(go.Bar(
        x=races,
        y=percentages,
        marker_color=colors,
        text=[f"{v}%" for v in percentages],
        textposition="outside"
    ))
    fig3.update_layout(
        yaxis=dict(title="Percentage (%)", range=[0, 75]),
        xaxis=dict(title="Race"),
        height=350,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444"),
        showlegend=False
    )
    fig3.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Graph 4: Radar chart overall comparison ───────────────
    st.subheader("Overall Model Comparison — Radar Chart")

    categories = ["Gender Accuracy", "Race Accuracy", "Age Score"]

    ml_radar = [81.21, 63.34, 100 - 12.15]
    dl_radar = [91.77, 76.14, 100 - 10.20]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=ml_radar + [ml_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="ML — Random Forest",
        line_color="#378ADD",
        fillcolor="rgba(55,138,221,0.15)"
    ))
    fig4.add_trace(go.Scatterpolar(
        r=dl_radar + [dl_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="DL — CNN",
        line_color="#1D9E75",
        fillcolor="rgba(29,158,117,0.15)"
    ))
    fig4.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        height=420,
        margin=dict(t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444")
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Age Score = 100 - MAE (higher is better). Radar chart shows CNN outperforms RF on all three tasks.")

    st.divider()
