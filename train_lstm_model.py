# train_lstm_model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

DATA_PATH = 'MP_Data'
SEQUENCE_LENGTH = 30

# Load data and labels
X, y, labels = [], [], []
label_map = {label: idx for idx, label in enumerate(os.listdir(DATA_PATH))}

for label in label_map:
    label_folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(label_folder):
        data = np.load(os.path.join(label_folder, file))
        if data.shape == (SEQUENCE_LENGTH, 42):
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = to_categorical(y).astype(int)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 42)))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_map), activation='softmax'))

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model and label map
model.save('lstm_model.h5')
np.save('label_map.npy', label_map)
print("✅ Model trained and saved.")
