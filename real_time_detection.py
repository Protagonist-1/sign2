import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Load label map and reverse it for decoding
label_map = np.load('label_map.npy', allow_pickle=True).item()
label_encoder = {v: k for k, v in label_map.items()}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Constants
SEQUENCE_LENGTH = 30
buffer = deque(maxlen=SEQUENCE_LENGTH)

# Start video capture
cap = cv2.VideoCapture(0)

print("🚀 Starting real-time detection... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip for mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract keypoints
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # 21 landmarks × 2 = 42 features
            buffer.append(landmarks)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Predict when we have enough frames
    if len(buffer) == SEQUENCE_LENGTH:
        sequence = np.expand_dims(buffer, axis=0)  # shape: (1, 30, 42)
        prediction = model.predict(sequence)[0]
        class_id = np.argmax(prediction)
        confidence = prediction[class_id]
        predicted_label = label_encoder[class_id]

        # Show prediction if confidence is high, else show in red
        color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
        cv2.putText(image, f'{predicted_label} ({confidence:.2f})', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        print(f'Predicted: {predicted_label}, Confidence: {confidence:.2f}')

    # Show frame
    cv2.imshow('Sign Detection (LSTM)', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
