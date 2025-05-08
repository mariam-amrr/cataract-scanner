import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd

# Load classification models
@st.cache_resource
def load_classification_models():
    model_eff = load_model("EfficientNetB0_final .keras")
    model_dense = load_model("DenseNet121_final .keras")
    return model_eff, model_dense

# Load YOLO model (for detection only)
@st.cache_resource
def load_yolo_model():
    return YOLO("best (4).pt")

model_eff, model_dense = load_classification_models()
yolo_model = load_yolo_model()

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



LOG_PATH = "user_log.csv"

def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    else:
        return pd.DataFrame(columns=["Name", "Prediction", "Confidence"])

def log_result(name, prediction, confidence):
    df = load_log()
    new_entry = pd.DataFrame([{
        "Name": name,
        "Prediction": prediction,
        "Confidence": round(confidence * 100, 2)
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


# --- UI ---
st.set_page_config(page_title="👁️ OcuScan", layout="wide")
# --- User name input ---

user_name = st.text_input("👤 Enter your name to begin:", placeholder="e.g., John Doe")

if not user_name:
    st.warning("Please enter your name to continue.")


st.title("👁️ Eye Disease Scanner")
st.markdown("Upload eye image to start analyzing")

# --- Session State Init ---
if "analyze_pressed" not in st.session_state:
    st.session_state.analyze_pressed = False
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = ""

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file and user_name:

    # If new image is uploaded, reset the analyze flag
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.analyze_pressed = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    img = Image.open(uploaded_file).convert('RGB')

    # Layout: image preview + analyze button
    upload_col, preview_col = st.columns([1, 2])

    with upload_col:
        if not st.session_state.analyze_pressed:
            st.image(img, caption='Preview', width=200)

    if not st.session_state.analyze_pressed:
        with preview_col:
            if st.button("🔍 Analyze Image"):
                st.session_state.analyze_pressed = True

    if st.session_state.analyze_pressed:

        # YOLO Detection
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            img.save(temp_file.name)
            temp_path = temp_file.name

        results = yolo_model.predict(source=temp_path, conf=0.1)
        boxes = results[0].boxes
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        os.remove(temp_path)

        # Classification
        img_eff = effnet_preprocess(preprocess_image(img))
        img_dense = densenet_preprocess(preprocess_image(img))

        pred_eff = model_eff.predict(img_eff)
        pred_dense = model_dense.predict(img_dense)
        ensemble_pred = (0.7 * pred_eff + 0.3 * pred_dense)
        predicted_class = class_names[np.argmax(ensemble_pred)]
        confidence = np.max(ensemble_pred)
        log_result(user_name, predicted_class, confidence)


        # --- RESULT BLOCK (2 COLUMNS) ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(" Cataract Detection")
            if boxes.cls.numel() > 0:
                detected_classes = [yolo_model.names[int(c)] for c in boxes.cls]
                st.success(f"Detected: {', '.join(detected_classes)}")
            else:
                st.warning("⚠️ No objects detected.")
            st.image(annotated_img, caption="Cataract Detection", width=300)

        with c2:
            st.subheader(" Classification Result")
            st.markdown(f"###  Class: **{predicted_class.upper()}**")
            st.markdown(f"📊 **Confidence:** `{confidence * 100:.2f}%`")

            with st.expander("🔬 Class Probabilities"):
                st.json({class_names[i]: float(ensemble_pred[0][i]) for i in range(len(class_names))})
# --- Editable Log Viewer ---
st.markdown("---")
st.subheader("🗂️ View & Edit Log Records")

log_df = load_log()

if not log_df.empty:
    edited_df = st.data_editor(log_df, num_rows="dynamic", use_container_width=True, key="editor")

    if st.button("💾 Save Changes to Log"):
        edited_df.to_csv(LOG_PATH, index=False)
        st.success("✅ Log updated successfully.")
else:
    st.info("📭 No log records available yet.")
import shutil

st.markdown("### 🗑️ Manage Logs")

if os.path.exists(LOG_PATH):
    if st.button("❌ Delete All Log Records"):
        os.remove(LOG_PATH)
        st.success("Log file deleted successfully. Refresh the app to update view.")
else:
    st.info("No log file found to delete.")


