# 🎵 InstruNet: Polyphonic Musical Instrument Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

**InstruNet** is a Deep Learning system designed to identify multiple musical instruments playing simultaneously (polyphony) in complex audio tracks. It utilizes a custom **Regularized Convolutional Neural Network (CNN)** trained on Mel-Spectrograms to achieve robust multi-label classification.

## 🚀 Key Features
* **Polyphonic Detection:** Can identify multiple instruments (e.g., "Voice + Guitar") in a single track.
* **Robust Architecture:** Custom CNN with **1024 filters**, L2 Regularization, and specialized `Conv -> BN -> ReLU` layering to prevent overfitting.
* **Sliding Window Analysis:** Processes long audio files by segmenting them into overlapping chunks and aggregating predictions.
* **Interactive Web UI:** A user-friendly **Streamlit** dashboard to upload audio and visualize Mel-Spectrograms and predictions in real-time.

## 🧠 Model Architecture
The core model (`vk_boosted_reg_model.keras`) is a highly tuned CNN designed for spectral analysis:
* **Input:** 128x128 Mel-Spectrograms.
* **Deep Feature Extraction:** 5 Convolutional Blocks scaling up to **1024 filters**.
* **Regularization:** * **L2 Weight Decay** on all Conv/Dense layers.
    * **Progressive Dropout** (0.3 -> 0.4 -> 0.5).
    * **Batch Normalization** applied *before* Activation for stability.
* **Global Average Pooling (GAP):** Reduces parameters and prevents spatial overfitting.

## 📊 Performance (IRMAS Test Set)
The model was evaluated on the **IRMAS** (Instrument Recognition in Musical Audio Signals) dataset.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Top-2 Accuracy** | **89.01%** | Probability that the correct instrument is in the top 2 guesses. |
| **Samples F1 Score** | **0.541** | Measures how accurately the full *set* of instruments is predicted per song. |
| **Macro F1 Score** | **0.441** | High score indicates good performance on rare instruments (e.g., Flute, Organ). |

*Note: The regularized model prioritizes generalization, resulting in a stable validation curve with zero overfitting.*

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/InstruNet.git](https://github.com/yourusername/InstruNet.git)
    cd InstruNet
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *(Key libraries: `tensorflow`, `librosa`, `streamlit`, `matplotlib`, `pandas`, `numpy`)*

3.  **Dataset Setup**
    * Download the [IRMAS Dataset](https://www.upf.edu/web/mtg/irmas).
    * Place the training data in `Spectrogram_Dataset/` and test audio in `test_audio/`.

## 💻 Usage

### 1. Run the Web Application
Launch the interactive dashboard to test the model on your own audio files.
```bash
streamlit run app.py
