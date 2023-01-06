# TechVidvan hand Gesture Recognizer

# import necessary packages

from operator import truediv
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from PIL import ImageFont, ImageDraw, Image
import pyautogui

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='./NFSans-Thin.otf',
                id=0)

# Initialize the webcam
cap = cv2.VideoCapture(0)
fontScale = 30
mainFont = ImageFont.truetype("./NFSansGX.ttf", fontScale)
font = cv2.FONT_HERSHEY_SIMPLEX
fontWeight = -1
b, g, r, a = 0, 0, 255, 0
tipIds = [4, 8, 12, 16, 20]
maxFontSize = 90
minFontSize = 10
maxFontWeight = 7
minFontWeight = -5
fontArray = ["./NFSans-Thin.otf", "./NFSans-Light.otf", "./NFSans-Regular.otf", "./NFSans-Medium.otf",
             "./NFSans-Bold.otf", "./NFSans-Heavy.otf"]
curFontWeight = 0
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framergb.flags.writeable = False
    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    # print('Handedness:', result.multi_handedness)
    className = ''

    changingFont = False
    changingSize = False
    # post process the result
    lmList = []
    handsType = []
    handType = ""
    if result.multi_hand_landmarks:
        for hand in result.multi_handedness:
            # print(hand)
            # print(hand.classification)
            # print(hand.classification[0])
            handType = hand.classification[0].label
            print(handType)
            handsType.append(handType)
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            print(className)

            # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 0, 255), 2, cv2.LINE_AA)
    mainFont = ImageFont.truetype("./NFSansGX.ttf", fontScale)
    if handType == "Left":
        if className == "thumbs up":
            if fontScale < maxFontSize:
                fontScale += 1
        elif className == "thumbs down":
            if fontScale > minFontSize:
                fontScale -= 1
        elif className == "live long" or className == "stop":
            if fontWeight < maxFontWeight:
                # fontWeight += 1
                if curFontWeight < len(fontArray) - 1:
                    curFontWeight += 1
                    ft.loadFontData(fontFileName=fontArray[curFontWeight],
                                    id=0)
        elif className == "fist":
            if fontWeight > minFontWeight:
                # fontWeight -= 1
                if curFontWeight >= 1:
                    curFontWeight -= 1
                    ft.loadFontData(fontFileName=fontArray[curFontWeight],
                                    id=0)
        elif className == "peace":
            image = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite("in_memory_to_disk.png", image)

    elif handType == "Right":
        if className == "stop":
            ft.loadFontData(fontFileName='./Futura.ttc',
                            id=0)
        elif className == "live long":
            ft.loadFontData(fontFileName='./NothingYouCouldDo-Regular.ttf',
                            id=0)
        elif className == "rock":
            ft.loadFontData(fontFileName='./Arbutus-Regular.ttf',
                            id=0)
        elif className == "fist":
            ft.loadFontData(fontFileName='./Modak-Regular.ttf',
                            id=0)
        elif className == "okay":
            ft.loadFontData(fontFileName='./RougeScript-Regular.ttf',
                            id=0)

    # cv2.putText(frame, "Hello there", (10, 30), font,
    #             fontScale, (0, 0, 255), fontWeight, cv2.LINE_AA)

    img2 = np.zeros((600, 600, 3), dtype=np.uint8)
    ft.putText(img=img2,
               text='HI THERE',
               org=(100, 300),
               fontHeight=fontScale,
               color=(255, 255, 255),
               thickness=fontWeight,
               line_type=cv2.LINE_AA,
               bottomLeftOrigin=True)

    cv2.imshow("Output", frame)
    # cv2.imshow("Output", img)
    cv2.imshow("Output3", img2)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
