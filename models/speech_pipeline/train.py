import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/processed/speech"

# =========================
# Load Data
# =========================

X = []
y = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        emotion = file.split("_")[0]
        features = np.load(os.path.join(DATA_PATH, file))
        X.append(features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)
print("Unique labels:", sorted(set(y)))
print("Number of classes:", len(set(y)))
print("Dataset shape:", X.shape)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =========================
# Build Model
# =========================

# =========================
# Build Model (Functional API)
# =========================

inputs = tf.keras.layers.Input(shape=(94, 144))

x = tf.keras.layers.Conv1D(64, 3, activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128)
)(x)

embedding = tf.keras.layers.Dense(128, activation="relu", name="speech_embedding")(x)

x = tf.keras.layers.Dropout(0.4)(embedding)

outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_speech_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# =========================
# Train
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)
# =========================
# Evaluate
# =========================

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
# =========================
# Save Embedding Model
# =========================

speech_embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("speech_embedding").output
)

speech_embedding_model.save("models/speech_pipeline/speech_embedding_model.keras")

print("Speech embedding model saved.")
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))