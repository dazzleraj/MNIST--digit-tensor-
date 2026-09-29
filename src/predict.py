import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, Dict
from PIL import Image
from src.preprocessing import preprocess_image


_CACHED_MODEL = None


def get_model(model_path: str = "models/mnist_cnn.keras") -> tf.keras.Model:
    """Loads and caches the trained Keras model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        _CACHED_MODEL = tf.keras.models.load_model(model_path)
    return _CACHED_MODEL


def predict_digit(
    image_input: Union[Image.Image, np.ndarray, str],
    model: tf.keras.Model = None,
    model_path: str = "models/mnist_cnn.keras"
) -> Tuple[int, float, Dict[int, float], np.ndarray]:
    """Preprocesses an input image and predicts the handwritten digit.

    Args:
        image_input: PIL Image, numpy array, or file path.
        model: Optional pre-loaded Keras model.
        model_path: Path to model if model is not passed.

    Returns:
        (predicted_digit, confidence_percent, class_probabilities, display_image_28x28)
    """
    if model is None:
        model = get_model(model_path)

    batch_tensor, display_img = preprocess_image(image_input)

    # Inference
    probabilities = model.predict(batch_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted_digit] * 100.0)

    class_probs = {digit: float(prob) for digit, prob in enumerate(probabilities)}

    return predicted_digit, confidence_pct, class_probs, display_img


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        digit, conf, probs, _ = predict_digit(img_path)
        print(f"Predicted Digit: {digit}")
        print(f"Confidence: {conf:.2f}%")
        print("Class Probabilities:")
        for d, p in probs.items():
            print(f"  {d}: {p*100:.2f}%")
    else:
        print("Usage: python src/predict.py <path_to_digit_image>")
