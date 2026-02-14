import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# Load Embedding Models
# =========================

speech_model = tf.keras.models.load_model(
    "models/speech_pipeline/speech_embedding_model.keras"
)

from transformers import TFBertModel, BertTokenizer

# Build BERT again
bert = TFBertModel.from_pretrained("bert-base-uncased")
bert.trainable = False

speech_model.trainable = False

# =========================
# Load Speech Features
# =========================

speech_data_path = "data/processed/speech"
raw_data_path = "data/raw/TESS"

speech_features = []
text_inputs = []
labels = []

for root, _, files in os.walk(raw_data_path):
    for file in files:
        if file.endswith(".wav"):

            folder_name = os.path.basename(root).lower()

            # Normalize emotion
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

            # Load speech feature
            npy_name = f"{emotion}_{file.replace('.wav', '.npy')}"
            speech_path = os.path.join(speech_data_path, npy_name)
            speech_feat = np.load(speech_path)
            speech_features.append(speech_feat)

            # Build text
            word = file.replace(".wav", "").split("_")[1]
            sentence = f"Say the word {word}"
            text_inputs.append(sentence)

            labels.append(emotion)

# Convert to arrays
speech_features = np.array(speech_features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

num_classes = len(le.classes_)

# =========================
# Generate Speech Embeddings
# =========================

speech_embeddings = speech_model.predict(speech_features)

# =========================
# Tokenize Text
# =========================

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    text_inputs,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors="tf"
)

# =========================
# Generate Text Embeddings
# =========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    text_inputs,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors="tf"
)

bert_outputs = bert(
    encodings["input_ids"],
    attention_mask=encodings["attention_mask"]
)

cls_token = bert_outputs.last_hidden_state[:, 0, :]

# Same dense layer structure as text pipeline
text_dense = tf.keras.layers.Dense(
    128,
    activation="relu"
)

text_embeddings = text_dense(cls_token)

text_embeddings = text_embeddings.numpy()

# =========================
# Concatenate Embeddings
# =========================

fusion_features = np.concatenate(
    [speech_embeddings, text_embeddings],
    axis=1
)

print("Fusion feature shape:", fusion_features.shape)

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    fusion_features,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# =========================
# Fusion Classifier
# =========================

fusion_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(fusion_features.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

fusion_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fusion_model.summary()

# =========================
# Train
# =========================

history = fusion_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)

# =========================
# Evaluate
# =========================

loss, acc = fusion_model.evaluate(X_test, y_test)
print("Fusion Test Accuracy:", acc)
