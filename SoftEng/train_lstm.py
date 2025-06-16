import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# 🗂️ Load data from gesture folders
DATA_PATH = "data"
sequences, labels = [], []

for gesture in os.listdir(DATA_PATH):
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.isdir(gesture_path):
        continue
    for filename in os.listdir(gesture_path):
        file_path = os.path.join(gesture_path, filename)
        sequence = np.load(file_path)
        sequences.append(sequence)
        labels.append(gesture)

X = np.array(sequences)
y = np.array(labels)

# 🔢 Encode labels (e.g., hello=0, thanks=1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# ✂️ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 🧠 Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 63)))  # 30 frames, 21 landmarks * 3 (x,y,z)
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 💾 Save best model
checkpoint = ModelCheckpoint("asl_dynamic_lstm.h5", save_best_only=True, monitor='val_accuracy', mode='max')

# 🏋️ Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[checkpoint])

# 💡 Save label encoder for later use
np.save("label_encoder.npy", label_encoder.classes_)
print("✅ Training complete. Model and labels saved.")
