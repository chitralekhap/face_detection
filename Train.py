import os
import numpy as np
import cv2 as cv

lbph = cv.face.LBPHFaceRecognizer_create()

def create_train(dataset_path, image_size=(220, 220)):
    faces = []
    labels = []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_count] = person_name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv.resize(img, image_size)  
                    faces.append(img_resized)
                    labels.append(label_count)
                else:
                    print(f"Image {img_name} at {img_path} could not be loaded.")
            except Exception as e:
                print(f"Skipping image {img_name}: {e}")

        label_count += 1

    if len(faces) == 0 or len(labels) == 0:
        print("No valid images found. Please check your dataset structure.")
        return [], [], {}

    return np.array(faces, dtype='object'), np.array(labels, dtype='int'), label_map


dataset_path = 'faces'  
faces, labels, label_map = create_train(dataset_path)

if len(faces) == 0 or len(labels) == 0:
    print("Training aborted due to missing or invalid data.")
else:
    print(f'Training Started. Total subjects: {len(label_map)}')

  
    faces = np.array([face.astype('uint8') for face in faces])
    lbph.train(faces, labels)
    lbph.save('TrainedLBPH.yml')


    with open('label_map.txt', 'w') as f:
        for label, name in label_map.items():
            f.write(f'{label}:{name}\n')

    print('Training Complete!')
