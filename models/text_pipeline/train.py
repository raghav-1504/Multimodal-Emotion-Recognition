import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

DATA_PATH = "data/raw/TESS"

texts = []
labels = []

for root, _, files in os.walk(DATA_PATH):
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

            # Extract word from filename
            parts = file.replace(".wav", "").split("_")
            word = parts[1]  # middle part

            sentence = f"Say the word {word}"

            texts.append(sentence)
            labels.append(emotion)

print("Total samples:", len(texts))
print("Example text:", texts[0])
# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

num_classes = len(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text_list):
    return tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="tf"
    )

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)
bert = TFBertModel.from_pretrained("bert-base-uncased")

input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

bert_output = bert(input_ids, attention_mask=attention_mask)
cls_token = bert_output.last_hidden_state[:, 0, :]

text_embedding = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="text_embedding"
)(cls_token)
dropout = tf.keras.layers.Dropout(0.3)(text_embedding)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)

model = tf.keras.Model(
    inputs=[input_ids, attention_mask],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
history = model.fit(
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
    },
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=16
)
loss, acc = model.evaluate(
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
    },
    y_test
)

print("Text Test Accuracy:", acc)
# =========================
# Save Text Embedding Model
# =========================

text_embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("text_embedding").output
)

text_embedding_model.save("models/text_pipeline/text_embedding_model.keras")

print("Text embedding model saved.")