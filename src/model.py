import tensorflow as tf
from tensorflow.keras import layers, models


def build_mnist_cnn(input_shape: tuple = (28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """Builds a Convolutional Neural Network (CNN) for MNIST digit classification.

    Architecture:
        Input: (28, 28, 1)
        -> Conv2D (32 filters, 3x3, relu, padding='same')
        -> MaxPooling2D (2, 2)
        -> Conv2D (64 filters, 3x3, relu, padding='same')
        -> MaxPooling2D (2, 2)
        -> Flatten
        -> Dense (128 units, relu)
        -> Dropout (0.5)
        -> Dense (10 units, softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="mnist_cnn")

    return model


if __name__ == "__main__":
    cnn_model = build_mnist_cnn()
    cnn_model.summary()
