import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load dataset
DATASET_PATH = "mudra_data"
with open(os.path.join(DATASET_PATH, "mudra_samples.json"), "r") as f:
    data = json.load(f)

labels = []
features = []
mudra_classes = list(data.keys())

for idx, mudra in enumerate(mudra_classes):
    for sample in data[mudra]:
        if len(sample) == 42:  # Only x, y (21 * 2 = 42)
            features.append(sample)
            labels.append(idx)

features = np.array(features)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(256, activation="relu", input_shape=(42,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(len(mudra_classes), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=75, batch_size=16, validation_data=(X_test, y_test))

# Save model and class labels
model.save(os.path.join(DATASET_PATH, "mudra_model.h5"))
with open(os.path.join(DATASET_PATH, "mudra_classes.json"), "w") as f:
    json.dump(mudra_classes, f)

print("✅ Training complete. Model and labels saved!")
