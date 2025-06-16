import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model and labels
model = load_model("asl_dynamic_lstm.h5")
labels = np.load("label_encoder.npy")

# MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Sequence buffer to collect 30 frames
sequence = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 features (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(landmarks)

            # Keep only the latest 30 frames
            if len(sequence) > 30:
                sequence.pop(0)

            # Predict if sequence is ready
            if len(sequence) == 30:
                input_data = np.array(sequence).reshape(1, 30, 63)
                prediction = model.predict(input_data, verbose=0)
                pred_label = labels[np.argmax(prediction)]
                cv2.putText(frame, f"{pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
    else:
        sequence = []

    cv2.imshow("ASL Dynamic Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
