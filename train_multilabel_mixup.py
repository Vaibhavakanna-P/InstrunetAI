import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import librosa
import os
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
AUDIO_PATH = './Cleaned_Audio_Dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
BATCH_SIZE = 32
EPOCHS = 30 # Increased epochs because the task is harder

# --- 1. MATH HELPER ---
def audio_to_spectrogram(y):
    """Converts loaded audio array to normalized spectrogram"""
    spectrogram = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    # Pad/Crop
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    return spectrogram_norm

# --- 2. MULTI-LABEL GENERATOR ---
class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, n_classes=11, mix_probability=0.6):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.mix_prob = mix_probability # 60% chance to mix files
        self.indexes = np.arange(len(self.file_paths))

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, ID in enumerate(indexes):
            # 1. Load Primary File
            file_a = self.file_paths[ID]
            label_a = self.labels[ID]
            wav_a, _ = librosa.load(file_a, sr=SR)
            
            # 2. Decision: Do we mix?
            if random.random() < self.mix_prob:
                # Pick a random second file
                rand_idx = random.randint(0, len(self.file_paths) - 1)
                file_b = self.file_paths[rand_idx]
                label_b = self.labels[rand_idx]
                wav_b, _ = librosa.load(file_b, sr=SR)
                
                # Combine Audio (Ensure same length)
                min_len = min(len(wav_a), len(wav_b))
                wav_mix = (wav_a[:min_len] + wav_b[:min_len]) / 2 # Average them
                
                # Combine Labels (Logical OR)
                label_mix = np.maximum(label_a, label_b)
                
                # Generate Spec from Mix
                spec = audio_to_spectrogram(wav_mix)
                final_label = label_mix
            else:
                # Solo (No mix)
                spec = audio_to_spectrogram(wav_a)
                final_label = label_a

            X[i,] = np.expand_dims(spec, axis=-1)
            y[i,] = final_label
            
        return X, y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# --- 3. TRAINING ---
def train_multilabel():
    print("--- Loading Data Map ---")
    file_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(AUDIO_PATH) if os.path.isdir(os.path.join(AUDIO_PATH, d))])
    num_classes = len(classes)
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(AUDIO_PATH, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        for f in files:
            file_paths.append(os.path.join(cls_path, f))
            label = np.zeros(num_classes)
            label[class_indices[cls]] = 1
            labels.append(label)
            
    X_train, X_val, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
    
    # Create Generators with Mixing enabled
    train_gen = MixupGenerator(X_train, y_train, batch_size=BATCH_SIZE, n_classes=num_classes, mix_probability=0.6)
    val_gen = MixupGenerator(X_val, y_val, batch_size=BATCH_SIZE, n_classes=num_classes, mix_probability=0.3) 

    print("\n--- Building Multi-Label Model ---")
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'), 
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Sigmoid allows multi-label classification (Piano=Yes, Flute=Yes)
        layers.Dense(num_classes, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy']) 
    
    # ✅ UPDATED: Saving as 'v2' to protect your original model
    checkpoint = callbacks.ModelCheckpoint('instrunet_multilabel_v2.keras', save_best_only=True, monitor='val_loss')

    print("--- Starting Training (This will take longer) ---")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
    print("\n✅ Saved: 'instrunet_multilabel_v2.keras'")

if __name__ == "__main__":
    train_multilabel()