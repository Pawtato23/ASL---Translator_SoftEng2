import cv2
import mediapipe as mp
import numpy as np
import os

GESTURE_LABEL = "go"  # Change this as needed
SAVE_PATH = f"data/{GESTURE_LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
recording = False
sequence = []
sample_count = 0

print("Press 's' to start recording a gesture (30 frames). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)
                print(f"Recording... Frame {len(sequence)}")

                if len(sequence) == 30:
                    np.save(f"{SAVE_PATH}/sample_{sample_count}.npy", np.array(sequence))
                    print(f"✅ Saved: sample_{sample_count}")
                    sample_count += 1
                    sequence = []
                    recording = False

    else:
        if recording:
            print("⚠️ Hand not detected — resetting sequence.")
            sequence = []
            recording = False

    cv2.imshow("Gesture Recorder", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        print("▶️ Starting recording. Hold your hand steady.")
        recording = True
        sequence = []

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
