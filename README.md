# Handwritten Digit Recognition using MNIST + CNN (TensorFlow & Streamlit)

An end-to-end Machine Learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) built in TensorFlow/Keras and served via an interactive Streamlit web application.

---

## 🚀 Features

* **CNN Architecture**: 2 Convolutional blocks with ReLU and MaxPooling, Dense layer with Dropout (0.5), and Softmax output.
* **Modern Package Management**: Uses `uv` for lightning-fast virtual environment setup and dependency installation.
* **Dual-Mode Interactive UI (`ui.py`)**:
  - **✍️ Draw Digit**: Interactive canvas to draw any digit (0–9) in real-time.
  - **📁 Upload Image**: Drag & drop or upload handwritten digit images (`.png`, `.jpg`).
* **Detailed Analytics & Visualization**:
  - Top predicted digit & percentage confidence score.
  - Full 10-class probability distribution bar chart.
  - Preprocessed 28×28 grayscale input inspection.
  - Training loss/accuracy curves and test set confusion matrix.

---

## 📁 Project Structure

```text
mnist-digit  (tensorflow)/
├── .venv/                      # Virtual environment (managed with uv)
├── models/
│   ├── mnist_cnn.keras         # Saved trained Keras model
│   ├── training_history.png    # Training loss & accuracy curves
│   └── confusion_matrix.png    # Confusion matrix on 10k test images
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # MNIST loading, cropping, centering & normalization
│   ├── model.py                # CNN model architecture definition
│   ├── train.py                # Training pipeline with early stopping & checkpoints
│   ├── evaluate.py             # Evaluation on test dataset & confusion matrix
│   └── predict.py              # Single/batch inference module
├── ui.py                       # Streamlit web application
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup (using `uv`)

1. **Create Virtual Environment**:
   ```powershell
   uv venv --python 3.11 .venv
   ```

2. **Activate Virtual Environment**:
   - On Windows:
     ```powershell
     .venv\Scripts\activate
     ```

3. **Install Dependencies**:
   ```powershell
   uv pip install -r requirements.txt --python .venv\Scripts\python.exe
   ```

---

## 🏋️ Training & Evaluation

### 1. Train the CNN Model
```powershell
.venv\Scripts\python.exe src/train.py
```
*Trains for up to 10 epochs with early stopping, saves the best weights to `models/mnist_cnn.keras`, and outputs `models/training_history.png`.*

### 2. Evaluate Model Performance
```powershell
.venv\Scripts\python.exe src/evaluate.py
```
*Evaluates the model against 10,000 MNIST test images, achieving ~99% accuracy, and outputs classification metrics and a confusion matrix heatmap.*

### 3. CLI Prediction
```powershell
.venv\Scripts\python.exe src/predict.py <path_to_image>
```

---

## 🌐 Running the Streamlit Web Application

To launch the web interface:

```powershell
.venv\Scripts\streamlit.exe run ui.py
```

Open your browser at `http://localhost:8501` to interactively draw digits or upload digit images!
