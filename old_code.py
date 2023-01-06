# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import ImageFont, ImageDraw, Image

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# ft = cv2.freetype.createFreeType2()
# ft.loadFontData(fontFileName='NFSansGX.ttf',
#                 id=0)

# Initialize the webcam
cap = cv2.VideoCapture(0)
fontScale = 30
mainFont = ImageFont.truetype("./NFSansGX.ttf", fontScale)
font = cv2.FONT_HERSHEY_SIMPLEX
fontWeight = 2
b, g, r, a = 0, 0, 255, 0

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
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

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    mainFont = ImageFont.truetype("./NFSansGX.ttf", fontScale)
    if className == "thumbs up":
        fontScale += 1
    elif className == "thumbs down":
        fontScale -= 1

    # elif className == "live long" or className == "stop":

    cv2.putText(frame, "Hello there", (10, 30), font,
                fontScale, (0, 0, 255), fontWeight, cv2.LINE_AA)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    # draw.text((30, 80), "Hello there", font=mainFont, fill=(b, g, r, a))
    # img = np.array(img_pil)

    cv2.imshow("Output", frame)
    # cv2.imshow("Output", img)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
