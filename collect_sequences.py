import cv2
import os
import numpy as np
import mediapipe as mp

# Setup
SEQUENCE_LENGTH = 30
SEQUENCES_PER_CLASS = 10
PHRASES = ['hello', 'thank_you', 'i_love_you']
DATA_PATH = os.path.join('MP_Data')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results, flip=False):
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            if flip:
                hand = [[1 - x, y] for x, y in hand]  # Flip along X-axis
            keypoints.append(np.array(hand).flatten())

    # Pad if only one or no hand detected
    if len(keypoints) == 0:
        return np.zeros(42)
    return keypoints[0][:42]  # Use only first hand detected (consistency)

cap = cv2.VideoCapture(0)

for phrase in PHRASES:
    input(f"Press Enter to start collecting the '{phrase}' phrase.")  # Wait for Enter key press
    for seq in range(SEQUENCES_PER_CLASS):
        sequence = []
        flipped_sequence = []
        print(f'Collecting {phrase}, Sequence {seq+1}/{SEQUENCES_PER_CLASS}')

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Normal + flipped
            keypoints = extract_keypoints(results)
            flipped_keypoints = extract_keypoints(results, flip=True)

            sequence.append(keypoints)
            flipped_sequence.append(flipped_keypoints)

            # Optional: Draw hand landmarks
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(image_bgr, f'Phrase: {phrase} | Frame: {frame_num+1}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Collecting', image_bgr)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Save sequence (original)
        save_path = os.path.join(DATA_PATH, phrase)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'{phrase}_{seq}'), sequence)

        # Save flipped sequence
        np.save(os.path.join(save_path, f'{phrase}_{seq}_flipped'), flipped_sequence)

cap.release()
cv2.destroyAllWindows()
