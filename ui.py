import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Import local modules
from src.predict import predict_digit, get_model
from src.train import train_model

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .prediction-card {
        background-color: #f0f7ff;
        border: 2px solid #1E88E5;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
    }
    .digit-highlight {
        font-size: 4.2rem;
        font-weight: 900;
        color: #0D47A1;
        margin: 0;
        line-height: 1;
    }
    .confidence-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2E7D32;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "models/mnist_cnn.keras"
HISTORY_JSON = "models/training_history.json"
METRICS_JSON = "models/evaluation_metrics.json"


@st.cache_resource
def load_cached_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def main():
    st.markdown('<div class="main-title">🔢 Handwritten Digit Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Powered by TensorFlow / Keras CNN & MNIST</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Status & Info")
        model_exists = os.path.exists(MODEL_PATH)

        if model_exists:
            st.success("Model is ready")
            model = load_cached_model()
        else:
            st.warning("No trained model found at `models/mnist_cnn.keras`.")
            if st.button("Train Model Now", use_container_width=True):
                with st.spinner("Training CNN on MNIST dataset..."):
                    train_model()
                    st.cache_resource.clear()
                    st.rerun()
            model = None

        st.divider()
        st.markdown("### CNN Architecture")
        st.code(
            """Input (28×28×1)
  ↓
Conv2D (32, 3×3, ReLU)
  ↓
MaxPooling (2×2)
  ↓
Conv2D (64, 3×3, ReLU)
  ↓
MaxPooling (2×2)
  ↓
Flatten
  ↓
Dense (128, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (10, Softmax)""",
            language="text"
        )

        st.divider()
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Draw digits centered in the canvas.
        - Use moderate stroke width (15-20px).
        - For uploaded images, good contrast between digit and background works best.
        """)

    if not model_exists or model is None:
        st.info("👋 Please train the model using the button in the sidebar or run `python src/train.py` from your terminal to begin.")
        return

    # Main area tabs
    tab_draw, tab_upload, tab_metrics = st.tabs(["✍️ Draw Digit", "📁 Upload Image", "📊 Model Metrics & Performance"])

    # --- TAB 1: DRAW DIGIT ---
    with tab_draw:
        col_canvas, col_result = st.columns([1, 1], gap="large")

        with col_canvas:
            st.subheader("Draw a digit (0–9)")
            stroke_width = st.slider("Brush Width", min_value=10, max_value=30, value=18, step=2)

            if CANVAS_AVAILABLE:
                canvas_result = st_canvas(
                    fill_color="#000000",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="mnist_canvas",
                    return_image_data=True,
                )
            else:
                st.error("streamlit-drawable-canvas is not installed. Please use the Upload Image tab.")
                canvas_result = None

        with col_result:
            if canvas_result is not None and canvas_result.image_data is not None:
                raw_data = canvas_result.image_data
                if np.max(raw_data[:, :, :3]) > 0 or np.max(raw_data[:, :, 3]) > 0:
                    try:
                        pred_digit, confidence, class_probs, display_28 = predict_digit(
                            raw_data, model=model
                        )
                        render_prediction_output(pred_digit, confidence, class_probs, display_28)
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                else:
                    st.info("✏️ Draw a digit inside the black canvas on the left to see live prediction.")

    # --- TAB 2: UPLOAD IMAGE ---
    with tab_upload:
        col_up_in, col_up_out = st.columns([1, 1], gap="large")

        with col_up_in:
            st.subheader("Upload handwritten digit image")
            uploaded_file = st.file_uploader(
                "Choose an image (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                key="file_uploader"
            )

            if uploaded_file is not None:
                user_img = Image.open(uploaded_file)
                st.image(user_img, caption="Uploaded Original Image", width=250)

        with col_up_out:
            if uploaded_file is not None:
                try:
                    pred_digit, confidence, class_probs, display_28 = predict_digit(
                        user_img, model=model
                    )
                    render_prediction_output(pred_digit, confidence, class_probs, display_28)
                except Exception as e:
                    st.error(f"Inference error: {e}")
            else:
                st.info("📤 Upload an image file on the left to test prediction.")

    # --- TAB 3: METRICS ---
    with tab_metrics:
        st.subheader("Model Evaluation & Training Metrics")

        # Training Curves
        if os.path.exists(HISTORY_JSON):
            with open(HISTORY_JSON, "r") as f:
                hist_data = json.load(f)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Accuracy over Epochs")
                df_acc = pd.DataFrame({
                    "Train Accuracy": hist_data.get("accuracy", []),
                    "Val Accuracy": hist_data.get("val_accuracy", [])
                })
                st.line_chart(df_acc)

            with col2:
                st.markdown("#### Loss over Epochs")
                df_loss = pd.DataFrame({
                    "Train Loss": hist_data.get("loss", []),
                    "Val Loss": hist_data.get("val_loss", [])
                })
                st.line_chart(df_loss)

        # Test Evaluation Metrics
        if os.path.exists(METRICS_JSON):
            with open(METRICS_JSON, "r") as f:
                metrics_data = json.load(f)

            st.divider()
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Test Accuracy", f"{metrics_data.get('test_accuracy', 0)*100:.2f}%")
            mcol2.metric("Test Loss", f"{metrics_data.get('test_loss', 0):.4f}")

            if "confusion_matrix" in metrics_data:
                st.markdown("#### Test Confusion Matrix (10,000 Test Images)")
                cm_df = pd.DataFrame(
                    metrics_data["confusion_matrix"],
                    index=[f"True {i}" for i in range(10)],
                    columns=[f"Pred {i}" for i in range(10)]
                )
                st.dataframe(cm_df, use_container_width=True)


def render_prediction_output(pred_digit: int, confidence: float, class_probs: dict, display_28: np.ndarray):
    """Renders prediction results, 28x28 feed, and probability chart."""
    st.subheader("🎯 Prediction Result")

    with st.expander("🔍 CNN Input View (Preprocessed 28×28 Grayscale)", expanded=False):
        st.image(display_28, width=140, clamp=True, caption="Normalized 28x28 Image")

    # Prediction Card
    st.markdown(f"""
    <div class="prediction-card">
        <p style="margin:0; font-size:1.1rem; color:#555;">Predicted Digit</p>
        <p class="digit-highlight">{pred_digit}</p>
        <p class="confidence-text">Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Class Probabilities (0–9)")
    df_probs = pd.DataFrame({
        "Digit": [str(d) for d in range(10)],
        "Probability (%)": [class_probs[d] * 100 for d in range(10)]
    })

    st.bar_chart(df_probs.set_index("Digit"), color="#1E88E5")


if __name__ == "__main__":
    main()
