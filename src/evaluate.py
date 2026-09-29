import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.preprocessing import load_and_preprocess_mnist


def evaluate_model(
    model_path: str = "models/mnist_cnn.keras",
    confusion_matrix_path: str = "models/confusion_matrix.png",
    metrics_json_path: str = "models/evaluation_metrics.json"
):
    """Evaluates the trained MNIST model on the test dataset."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    _, (x_test, y_test) = load_and_preprocess_mnist()

    print("\n--- Running Evaluation on 10,000 MNIST Test Images ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%\n")

    # Generate predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("--- Detailed Classification Report ---")
    target_names = [f"Digit {i}" for i in range(10)]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save metrics to JSON
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    # Plot Confusion Matrix if matplotlib/seaborn are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title(f'MNIST Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)', fontsize=14)
        plt.xlabel('Predicted Digit', fontsize=12)
        plt.ylabel('True Digit', fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)
        plt.savefig(confusion_matrix_path, dpi=300)
        plt.close()
        print(f"Confusion matrix heatmap saved to {confusion_matrix_path}")
    except Exception as e:
        print(f"Note: Confusion matrix plot skipped ({e}). Metrics are saved in JSON.")

    return metrics


if __name__ == "__main__":
    evaluate_model()
