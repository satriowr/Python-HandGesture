import cv2
import mediapipe as mp #using mediapipe

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) #camera input

while True :
    ret, img, = cap.read()
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks :
        for hand_landmark in results.multi_hand_landmarks :
            for id, lm in enumerate(hand_landmark.landmark) :
                print(id, ":", lm)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

    cv2.imshow("Hand Track", img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
