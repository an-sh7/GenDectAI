from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import pygame
import time

# load model
model = load_model('gender_detection.model')

# initialize Pygame mixer
pygame.mixer.init()

# open webcam
webcam = cv2.VideoCapture(0)

# load sounds
male_sound = pygame.mixer.Sound('man.wav')
female_sound = pygame.mixer.Sound('woman.wav')

classes = ['man', 'woman']

last_played = time.time() - 5  # initialize to 5 seconds ago

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    faces, confidences = cv.detect_face(frame)

    # loop through detected faces
    for idx, (face, confidence) in enumerate(zip(faces, confidences)):

        # get corner points of face rectangle
        startX, startY, endX, endY = face

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        label = classes[np.argmax(conf)]
        conf = conf[np.argmax(conf)]

        label = "{}: {:.2f}%".format(label, conf * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # play sound based on gender if at least 1 seconds have passed since the last time
        if label.startswith('man') and time.time() - last_played >= 1:
            male_sound.play()
            print("Male Face Found")
            last_played = time.time()
        elif label.startswith('woman') and time.time() - last_played >= 1:
            female_sound.play()
            print("Female Face Found")
            last_played = time.time()

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
