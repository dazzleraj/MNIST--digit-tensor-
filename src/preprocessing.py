import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from typing import Tuple, Union


def load_and_preprocess_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads MNIST dataset, normalizes pixel values to [0, 1], and reshapes for CNN.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dimensions to (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def center_and_fit_image(pil_img: Image.Image, target_size: int = 28, padding: int = 4) -> Image.Image:
    """Crops the non-zero bounding box of the digit and fits it inside target_size x target_size

    with appropriate padding, preserving aspect ratio, centered like MNIST.
    """
    img_arr = np.array(pil_img)
    # Find bounding box of content (pixels > threshold)
    threshold = 30
    coords = np.argwhere(img_arr > threshold)

    if coords.size == 0:
        # Blank image
        return pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Crop to digit
    cropped = pil_img.crop((x0, y0, x1, y1))
    crop_w, crop_h = cropped.size

    # Fit into (target_size - 2*padding)
    inner_size = target_size - (2 * padding)
    if crop_w > crop_h:
        new_w = inner_size
        new_h = max(1, int(round((crop_h / crop_w) * inner_size)))
    else:
        new_h = inner_size
        new_w = max(1, int(round((crop_w / crop_h) * inner_size)))

    resized_digit = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create empty black canvas
    canvas = Image.new("L", (target_size, target_size), color=0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized_digit, (paste_x, paste_y))

    return canvas


def preprocess_image(
    image_input: Union[Image.Image, np.ndarray, str],
    invert_if_light: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses an input image into the format expected by the MNIST CNN.

    Args:
        image_input: PIL Image, NumPy array (grayscale/RGB/RGBA), or filepath.
        invert_if_light: If average corner intensity is bright, invert so background is black.

    Returns:
        (batch_tensor, display_array) where:
            - batch_tensor has shape (1, 28, 28, 1) and dtype float32 in [0, 1]
            - display_array has shape (28, 28) and dtype float32 in [0, 1]
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            # Handle RGBA from canvas
            # Alpha channel might have the drawing, or RGB might
            # If drawing is white on black transparent:
            alpha = image_input[:, :, 3]
            rgb = image_input[:, :, :3]
            if np.mean(alpha) > 0 and np.max(rgb) == 0:
                # Alpha contains drawing
                img = Image.fromarray(alpha)
            else:
                img = Image.fromarray(image_input).convert("RGB")
        else:
            img = Image.fromarray(image_input.astype(np.uint8))
    else:
        img = image_input

    # Convert to grayscale
    img_gray = img.convert("L")

    img_arr = np.array(img_gray, dtype=np.float32)

    # Auto-detect if background is light and digit is dark
    # Check 4 corners or average intensity
    corners = [
        img_arr[:5, :5],
        img_arr[:5, -5:],
        img_arr[-5:, :5],
        img_arr[-5:, -5:]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])

    if invert_if_light and corner_mean > 127:
        # Invert: white background -> black background
        img_gray = ImageOps.invert(img_gray)

    # Center and fit to 28x28
    processed_img = center_and_fit_image(img_gray, target_size=28, padding=4)

    # Convert to float array [0, 1]
    final_arr = np.array(processed_img, dtype=np.float32) / 255.0

    # Reshape for CNN
    batch_tensor = np.expand_dims(final_arr, axis=(0, -1))  # (1, 28, 28, 1)

    return batch_tensor, final_arr
