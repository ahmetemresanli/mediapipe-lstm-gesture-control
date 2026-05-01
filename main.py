import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
min_detection_confidence = 0.7 #eşik değeri
hands = mpHand.Hands() #bu metodun inputlarına bak (Hand())

mpDraw = mp.solutions.drawing_utils #el koordinatlarına göre çizim yapar

pTime = 0
cTime = 0

while True:
    success, img = cap.read() #success :görüntüyü alıp alamadığımızı gösteriyor
                              # img :kameradan read ettiğimiz görüntü

    if not success:
        print("Kamera bulunamadı !")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape #h:height, w:width, c:colour

                cx, cy = int(lm.x * w), int(lm.y * h)

                #bilek
                if id == 0:
                    cv2.circle(img, (cx, cy), 9, 255, cv2.FILLED)

    #fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS :"+ str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0,0,0), 3)

    cv2.imshow('Video', img)
    cv2.waitKey(1) #ne kadar süre bekleyeceğiz