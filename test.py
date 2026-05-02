import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

actions = ["play_pause", "volume_up", "volume_down", "next", "previous", "full_screen"]
sequence_length = 40
threshold = 0.80

model = tf.saved_model.load("model/gesture_saved_model")
infer = model.signatures["serving_default"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

sequence = []
last_action_time = 0
cooldown = 1.0


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


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kamera hatası")
        break

    display_text = "none"
    color = (0, 0, 255)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    landmarks = extract_landmarks(results)

    sequence.append(landmarks)
    sequence = sequence[-sequence_length:]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

            output = infer(tf.constant(input_data))
            prediction = list(output.values())[0].numpy()[0]

            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]
            predicted_action = actions[predicted_index]

            if confidence > threshold:
                display_text = f"{predicted_action} ({confidence:.2f})"
                color = (0, 255, 0)

                current_time = time.time()

                if current_time - last_action_time > cooldown:
                    print(f"Tahmin: {predicted_action} | Guven: {confidence:.2f}")
                    last_action_time = current_time

    cv2.putText(
        image,
        display_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Gesture Control", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()