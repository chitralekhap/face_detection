import cv2 as cv
import numpy as np
import os

lbph = cv.face.LBPHFaceRecognizer_create()
lbph.read('TrainedLBPH.yml')

label_map = {}
with open('label_map.txt', 'r') as f:
    for line in f.readlines():
        label, name = line.strip().split(':')
        label_map[int(label)] = name

face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')

def recognize_faces_in_folder(test_folder):
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        img = cv.imread(img_path)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            face_region = gray_img[y:y + h, x:x + w]
            label, confidence = lbph.predict(face_region)

            if confidence < 500:  
                name = label_map.get(label, "Unknown")
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(img, f'{name} ({int(confidence)}%)', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imshow('Recognized Image', img)
        cv.waitKey(0)  
        cv.destroyAllWindows()


test_folder = 'test_images'  
recognize_faces_in_folder(test_folder)
