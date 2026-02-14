import os
import librosa
import numpy as np
from tqdm import tqdm

RAW_PATH = "data/raw/TESS"
PROCESSED_PATH = "data/processed/speech"

SAMPLE_RATE = 16000
TARGET_DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 64


def pad_or_truncate(signal, sr, target_duration):
    target_length = int(sr * target_duration)

    if len(signal) > target_length:
        return signal[:target_length]
    else:
        padding = target_length - len(signal)
        return np.pad(signal, (0, padding), mode="constant")


def extract_features(filepath):
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    # Trim silence
    signal, _ = librosa.effects.trim(signal)

    # Fix length
    signal = pad_or_truncate(signal, SAMPLE_RATE, TARGET_DURATION)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC
    )

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_mels=N_MELS
    )

    # Convert to log scale
    mel = librosa.power_to_db(mel)

    # Concatenate features
    features = np.vstack([mfcc, delta, mel])

    return features.T  # (time_steps, feature_dim)


def process_dataset():
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    for root, _, files in os.walk(RAW_PATH):
        for file in tqdm(files):
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)

                # Emotion label
                folder_name = os.path.basename(root).lower()

                if "angry" in folder_name:
                    emotion = "angry"
                elif "disgust" in folder_name:
                    emotion = "disgust"
                elif "fear" in folder_name:
                    emotion = "fear"
                elif "happy" in folder_name:
                    emotion = "happy"
                elif "neutral" in folder_name:
                    emotion = "neutral"
                elif "sad" in folder_name:
                    emotion = "sad"
                elif "pleasant" in folder_name or "surprise" in folder_name:
                    emotion = "pleasant_surprise"
                else:
                    continue

                features = extract_features(filepath)

                save_path = os.path.join(
                    PROCESSED_PATH,
                    f"{emotion}_{file.replace('.wav', '.npy')}"
                )

                np.save(save_path, features)


if __name__ == "__main__":
    process_dataset()