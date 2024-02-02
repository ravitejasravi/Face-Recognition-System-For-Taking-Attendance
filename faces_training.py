import os
import cv2
import numpy as np
from PIL import Image
import pickle

# getting the project path
project_dir = os.path.dirname(os.path.abspath(__file__))

# joining the project path with images
images_dir = os.path.join(project_dir, "images")

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_label = []
x_train = []

# root- project path with images,
# dirs - it will print the image files in list
# files - it will print the images names
for root, dirs, files in os.walk(images_dir):
    # print(root)
    # print(dirs)
    # print(files)

    for img in files:
        if img.endswith("png") or img.endswith("jpg") or img.endswith("jpeg"):
            path = os.path.join(root, img)  # it will get the path of the images
            label = os.path.basename(root).replace(" ", "_").lower()  # only gets the folder name in images
            print(label, path)

            # assigning number to the labes or folder which contaings actual images
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            pil_image = Image.open(path).convert("L")  # getting the image and converting into gray
            image_array = np.array(pil_image, "uint8")  # generating array in an image
            # print(image_array)

            faces_in_pic = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for x, y, w, h in faces_in_pic:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_label.append(id_)

with open("label.pickle", 'wb') as label_fold:
    pickle.dump(label_ids, label_fold)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainerr.yml")
