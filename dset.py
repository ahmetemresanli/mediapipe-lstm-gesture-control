import cv2
import mediapipe as mp
import numpy as np
import os

DATA_PATH = "D:\mediapipe-lstm-gesture-control\dataset"

actions = ["play_pause", "volume_up", "volume_down", "next", "previous", "full_screen"]

sequence_length = 40
num_sequences = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


# Landmark çıkarma fonksiyonu
def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []

        wrist = hand.landmark[0]

        for lm in hand.landmark:
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        return np.array(landmarks)

    return np.zeros(63)

print("Hareket listesi:")
for i, action in enumerate(actions):
    print(f"{i}: {action}")

selected_idx = int(input())
selected_action = actions[selected_idx]

os.makedirs(os.path.join(DATA_PATH, selected_action), exist_ok=True)

cap = cv2.VideoCapture(0)

sequence = 0

while sequence < num_sequences:
    frames = []
    recording = False
    frame_num = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kamera hatasi")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        landmarks = extract_landmarks(results)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Kayıt başlamadıysa
        if not recording:
            cv2.putText(
                image,
                f"{selected_action} | Ornek: {sequence}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                image,
                "Baslamak icin 's' tusu",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        # Kayıt başladıysa
        else:
            frames.append(landmarks)

            cv2.putText(
                image,
                f"KAYIT | Frame: {frame_num}/{sequence_length}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            frame_num += 1

            if frame_num == sequence_length:
                save_path = os.path.join(DATA_PATH, selected_action, f"{sequence}.npy")
                np.save(save_path, np.array(frames))

                print(f"Kaydedildi: {save_path}")

                sequence += 1
                break

        cv2.imshow("Veri Toplama", image)

        key = cv2.waitKey(10) & 0xFF

        # Kayıt başlat
        if key == ord("s") and not recording:
            recording = True
            frames = []
            frame_num = 0
            print("Kayıt başladı")

        # Çıkış
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()