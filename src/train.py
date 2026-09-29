import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from src.preprocessing import load_and_preprocess_mnist
from src.model import build_mnist_cnn


def plot_history(history_dict: dict, save_path: str):
    """Plots training and validation accuracy & loss curves if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        # Accuracy subplot
        plt.subplot(1, 2, 1)
        plt.plot(history_dict.get('accuracy', []), label='Train Accuracy', marker='o')
        plt.plot(history_dict.get('val_accuracy', []), label='Val Accuracy', marker='s')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history_dict.get('loss', []), label='Train Loss', marker='o')
        plt.plot(history_dict.get('val_loss', []), label='Val Loss', marker='s')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training history plot saved to {save_path}")
    except Exception as e:
        print(f"Note: Matplotlib plotting skipped ({e}). History saved as JSON.")


def train_model(
    epochs: int = 10,
    batch_size: int = 64,
    model_save_path: str = "models/mnist_cnn.keras",
    history_plot_path: str = "models/training_history.png",
    history_json_path: str = "models/training_history.json"
) -> tf.keras.Model:
    """Trains the MNIST CNN model and saves the trained weights."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("--- 1. Loading MNIST Dataset ---")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    print("\n--- 2. Building CNN Model ---")
    model = build_mnist_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    print("\n--- 3. Training Model ---")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model explicitly
    model.save(model_save_path)
    print(f"\nTrained model successfully saved to: {model_save_path}")

    # Save history as JSON
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_json_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to {history_json_path}")

    # Plot and save training history
    plot_history(history_serializable, history_plot_path)

    # Quick test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nImmediate Test Set Evaluation -> Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

    return model


if __name__ == "__main__":
    train_model()
