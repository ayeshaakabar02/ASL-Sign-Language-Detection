import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import base64
import datetime
import plotly.graph_objs as go
import pandas as pd
from eda import run_eda
from game import run_game
import tensorflow as tf
from streamlit_option_menu import option_menu
from auth import authenticate, signup
from auth_ui import login_signup_page
from footer import add_footer




# === LOGIN PAGE ===
login_signup_page()

# --- Setup page ---
st.set_page_config(page_title="ASL Sign Language Detection", page_icon="images/favicon.png", layout="wide")

# --- Background Image ---
def set_background(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("images/background.jpg")

# --- Session and Model Usage Logging ---
if "logs" not in st.session_state:
    st.session_state.logs = []
    st.session_state.model_usage = {}
    st.session_state.prediction_events = []
    st.session_state.feedback = []

st.session_state.logs.append(f"Visited at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === APP STARTS ===
class_labels = sorted(os.listdir("data/train"))

model_paths = {
    "MobileNetV2": "mobilenetv2_best.h5",
    "NASNetMobile": "nasnetmobile_best.h5",
    "Xception": "xception_best.h5"
}

# === Setup Navigation ===
page = option_menu(
    menu_title=None,
    options=["Home", "Image Prediction", "Webcam Capture", "Model Accuracy", "EDA Dashboard", "Admin Dashboard", "Guess the Sign Game", "Logout"],
    icons=["house", "image", "camera", "bar-chart", "bar-chart-line", "tools", "controller","box-arrow-right"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0d388e"},
        "icon": {"color": "black", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "margin": "0 10px", "--hover-color": "#cce7ff"},
        "nav-link-selected": {"background-color": "#919baa", "color": "white"},
    }
)

@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    image = np.asarray(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(model, img_tensor):
    predictions = model.predict(img_tensor)
    class_index = np.argmax(predictions)
    return class_labels[class_index]

if page == "Home":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/logo.png", width=150)

    with col2:
        st.title("🧠 Automatic Sign Language Detection")
        st.markdown("""
        This application uses **Deep Learning** to detect American Sign Language (ASL) signs using webcam or uploaded images.
        You can explore model accuracy, predictions, real-time detection, and interactive games.
        """)

    st.markdown("---")
    st.subheader("📸 Sample Input Images")

    image_paths = ["images/sample1.jpg", "images/sample2.webp", "images/sample3.webp"]
    output_size = (224, 224)

    # Resize and show in columns
    cols = st.columns(3)
    for i, path in enumerate(image_paths):
                      
        try:
          img = Image.open(path).convert("RGB")
          img = ImageOps.fit(img, output_size, Image.Resampling.LANCZOS)
          cols[i].image(img, caption=f"Sample {i+1}", use_container_width=True)

        except Exception as e:
         cols[i].error(f"❌ Failed to load {path}\n{e}")

 
    st.markdown("""
    ### 🔍 Features:
    - Upload an image or use your **webcam** to detect signs in real-time.
    - Analyze model **performance**, confusion matrix, and class-level accuracy.
    - Play the **'Guess the Sign'** game to test your skills!

    ### 🧠 Models Used:
    - ✅ MobileNetV2
    - ✅ NASNetMobile
    - ✅ Xception

    ### 📦 Dataset:
    - American Sign Language (28 classes: A-Z, 'space', 'nothing')
    """)
elif page == "Image Prediction":
    st.title("🖼️ Upload Image for Prediction")
 
    model_choice = st.selectbox("Choose a model", list(model_paths.keys()))
    model = load_model_cached(model_paths[model_choice])
    st.success(f"✅ {model_choice} model loaded")

    st.session_state.model_usage[model_choice] = st.session_state.model_usage.get(model_choice, 0) + 1

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_tensor = preprocess_image(uploaded_file)
        predictions = model.predict(image_tensor)
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = float(predictions[0][predicted_index])
        loss = 1.0 - confidence

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success(f"🧠 Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")

  


        st.markdown("### 📉 Prediction Confidence vs. Loss")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Confidence"], y=[confidence * 100], name="Confidence", marker_color='green'))
        fig.add_trace(go.Bar(x=["Loss"], y=[loss * 100], name="Loss", marker_color='red'))
        fig.update_layout(yaxis_title="Percentage (%)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔍 Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Confidence (%)": (predictions[0] * 100).round(2)
        }).sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df)

        true_class = st.selectbox("Select the true label (optional for feedback)", class_labels)

        if st.button("Compare with Prediction"):
            is_correct = (predicted_class == true_class)
            st.markdown(f"🎯 Match: **{is_correct}**")
            cm_data = {
                "Actual Label": [true_class],
                "Predicted Label": [predicted_class]
            }
            cm_df = pd.DataFrame(cm_data)
            fig_cm = go.Figure(data=go.Table(
                header=dict(values=list(cm_df.columns), fill_color='lightblue', align='left'),
                cells=dict(values=[cm_df["Actual Label"], cm_df["Predicted Label"]],
                           fill_color='lavender', align='left')
            ))
            st.markdown("### 🔍 Confusion Matrix (Single Prediction)")
            st.plotly_chart(fig_cm, use_container_width=True)

        feedback = st.radio("Was the prediction correct?", ["Yes", "No"], key="feedback_img")
        if st.button("Submit Feedback"):
            st.session_state.feedback.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_choice,
                "prediction": predicted_class,
                "confidence": confidence * 100,
                "true_label": true_class,
                "feedback": feedback
            })
            st.success("✅ Feedback saved. Thank you!")




        elif page == "Word Prediction":
         st.title("Word Prediction")
    st.write("Upload a sequence of images to predict a word")

    # Get the alphabet sequence
    alphabet_sequence = []
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_tensor = preprocess_image(uploaded_file)
            predictions = model.predict(image_tensor)
            predicted_index = np.argmax(predictions)
            alphabet = class_labels[predicted_index]
            alphabet_sequence.append(alphabet)

        # Predict the word
        if alphabet_sequence:
            predicted_word = "".join(alphabet_sequence)
            st.write(f"Predicted word: {predicted_word}")




elif page == "Webcam Capture":

    st.title("📷 Real-Time ASL Detection via Webcam")
    selected_model = st.selectbox("Select a model", list(model_paths.keys()))
    model = load_model_cached(model_paths[selected_model])  # <-- model loaded here
    st.success(f"✅ {selected_model} model loaded")

    st.session_state.model_usage[selected_model] = st.session_state.model_usage.get(selected_model, 0) + 1

    start_cam = st.checkbox("Start Webcam")
    stop_button = st.button("🛑 Stop Webcam")

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    if start_cam and not stop_button:
        frame_placeholder = st.empty()
        prediction_placeholder = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            predicted_class = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min = min([lm.x for lm in hand_landmarks.landmark])
                    y_min = min([lm.y for lm in hand_landmarks.landmark])
                    x_max = max([lm.x for lm in hand_landmarks.landmark])
                    y_max = max([lm.y for lm in hand_landmarks.landmark])

                    x1, y1 = int(x_min * w), int(y_min * h)
                    x2, y2 = int(x_max * w), int(y_max * h)

                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi = cv2.resize(roi, (224, 224))
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        img_array = np.expand_dims(roi_rgb / 255.0, axis=0)

                        # ✅ This is where error was occurring, now model is defined
                        predictions = model.predict(img_array)
                        predicted_class = class_labels[np.argmax(predictions)]

                        st.session_state.prediction_events.append({
                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": selected_model,
                            "prediction": predicted_class
                        })

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{predicted_class}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            prediction_placeholder.info(
                f"✋ Detected: **{predicted_class}**" if predicted_class else "👀 Waiting for hand...")

            if stop_button:
                break

        cap.release()
        frame_placeholder.empty()
        prediction_placeholder.empty()
        st.warning("🛑 Webcam Stopped")




elif page == "Model Accuracy":

    st.title("📈 Model Accuracy & Loss Charts")
    tabs = st.tabs(list(model_paths.keys()))

    model_histories = {
        "MobileNetV2": ([0.81, 0.95, 0.96, 0.97, 0.98], [0.88, 0.20, 0.15, 0.13, 0.10]),
        "NASNetMobile": ([0.72, 0.86, 0.89, 0.91, 0.92], [1.43, 0.65, 0.43, 0.33, 0.28]),
        "Xception": ([0.87, 0.99, 0.98, 0.99, 0.99], [0.77, 0.03, 0.05, 0.02, 0.03])
    }

    for i, model in enumerate(model_paths.keys()):
        with tabs[i]:
            acc, loss = model_histories[model]
            st.subheader(f"📊 {model} - Accuracy and Loss")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=acc, mode='lines+markers', name='Accuracy'))
            fig.add_trace(go.Scatter(y=loss, mode='lines+markers', name='Loss'))
            fig.update_layout(title=f"{model} Training History", xaxis_title="Epoch", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

elif page == "EDA Dashboard":
    run_eda("data/train")


elif page == "Admin Dashboard":
  
    st.title("🔧 Admin Dashboard")

    st.subheader("📂 Session Logs")
    st.write("These logs show when the app was accessed during this session:")
    st.code("\n".join(st.session_state.logs))

    st.subheader("📊 Model Usage Frequency")
    if st.session_state.model_usage:
        usage_df = pd.DataFrame.from_dict(st.session_state.model_usage, orient='index', columns=['Usage Count'])
        fig_usage = go.Figure([go.Bar(x=usage_df.index, y=usage_df["Usage Count"], marker_color='indianred')])
        fig_usage.update_layout(title="Model Usage Count", xaxis_title="Model", yaxis_title="Count")
        st.plotly_chart(fig_usage, use_container_width=True)
    else:
        st.info("No model usage recorded yet.")

    st.subheader("📝 Prediction Feedback")
    if st.session_state.feedback:
        df_fb = pd.DataFrame(st.session_state.feedback)
        st.dataframe(df_fb)
    else:
        st.info("No feedback submitted yet.")

    st.subheader("📡 Real-Time Prediction Events")
    if st.session_state.prediction_events:
        df_events = pd.DataFrame(st.session_state.prediction_events)
        st.dataframe(df_events)
    else:
        st.info("No predictions made yet.")

    st.subheader("📥 Export Logs")
    if st.button("Download Logs as CSV"):
        logs_df = pd.DataFrame({"logs": st.session_state.logs})
        logs_csv = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", logs_csv, "logs.csv", "text/csv")

    if st.button("Download Feedback as Excel"):
        if st.session_state.feedback:
            fb_df = pd.DataFrame(st.session_state.feedback)
            excel_bytes = fb_df.to_excel(index=False, engine='openpyxl')
            st.download_button("Download Excel", excel_bytes, "feedback.xlsx")
        else:
            st.warning("No feedback to export.")

elif page == "Guess the Sign Game":
    run_game()

elif page == "Logout":
    st.title("🔒 Logout")
    st.write("Click below to logout from your session.")

    if st.button("Logout"):
        from auth_ui import logout_user
        logout_user()


    # === Footer ===
add_footer()

