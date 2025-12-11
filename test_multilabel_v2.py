import tensorflow as tf
import numpy as np
import librosa
import os
import glob

# --- CONFIGURATION ---
# Point this to your NEW model
MODEL_PATH = 'instrunet_multilabel_v2.keras' 
TEST_FOLDER = './Test_Audio'
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0 

# Must match training alphabetical order
CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def get_ground_truth(txt_path):
    with open(txt_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def process_chunk(y_chunk):
    """
    Exact math from train_multilabel_mixup.py
    """
    spectrogram = librosa.feature.melspectrogram(y=y_chunk, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    return spectrogram_norm

def predict_song(model, wav_path):
    try:
        y_full, _ = librosa.load(wav_path, sr=SR, mono=True)
        y_full = librosa.util.normalize(y_full)
    except:
        return None

    total_samples = len(y_full)
    chunk_samples = int(CHUNK_DURATION * SR)
    hop_length = chunk_samples 
    
    predictions = []

    for start_idx in range(0, total_samples, hop_length):
        end_idx = start_idx + chunk_samples
        chunk = y_full[start_idx:end_idx]
        
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        spec = process_chunk(chunk)
        input_batch = np.expand_dims(spec, axis=0)
        input_batch = np.expand_dims(input_batch, axis=-1)
        
        pred = model.predict(input_batch, verbose=0)[0]
        predictions.append(pred)

    if not predictions: return None
    
    # Average the probabilities across the whole song
    return np.mean(predictions, axis=0)

def run_evaluation():
    print(f"--- Loading New Model: {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Did training finish?")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    wav_files = glob.glob(os.path.join(TEST_FOLDER, "*.wav"))
    
    top1_correct = 0
    total_count = 0
    
    print(f"{'FILENAME':<25} | {'TOP GUESS':<10} | {'TRUE LABELS':<15} | {'RESULT'}")
    print("-" * 70)

    for wav_path in wav_files:
        txt_path = wav_path.replace('.wav', '.txt')
        if not os.path.exists(txt_path): continue
            
        true_labels = get_ground_truth(txt_path)
        avg_probs = predict_song(model, wav_path)
        if avg_probs is None: continue
        
        # --- METRIC 1: Top-1 Accuracy (Comparison with Old Model) ---
        top_index = np.argmax(avg_probs)
        top_guess = CLASS_NAMES[top_index]
        confidence = avg_probs[top_index] * 100
        
        is_correct = top_guess in true_labels
        
        if is_correct:
            top1_correct += 1
            icon = "✅"
        else:
            icon = "❌"
            
        total_count += 1
        
        # Display
        fname = os.path.basename(wav_path)[:23]
        print(f"{fname:<25} | {top_guess} ({int(confidence)}%) | {','.join(true_labels):<15} | {icon}")

    if total_count > 0:
        acc = (top1_correct / total_count) * 100
        print("-" * 70)
        print(f"📊 NEW MODEL ACCURACY (Top-1): {acc:.2f}%")
        print(f"   (Old Robust Model was ~68.9%)")
    else:
        print("❌ No test files found.")

if __name__ == "__main__":
    run_evaluation()