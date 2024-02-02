import numpy as np
import cv2
import pickle

# fbfgg = np

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainerr.yml")

labels = {"person_name": 1}
with open("label.pickle", 'rb') as label_fold:
    original_label = pickle.load(label_fold)
    labels = {v:k for k, v in original_label.items()}

capture = cv2.VideoCapture(0)

while True:
    # capturing frame by frame
    ret, frame = capture.read()

    # converting video to gray, cvt - convert
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting frames in a video
    faces_in_video = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces_in_video:
        # print(x, y, w, h)
        part_gray = gray[y:y + h, x:x + w]  # capturing exactly face

        id_, accuracy = recognizer.predict(part_gray)
        if accuracy >= 45 or accuracy <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_ITALIC
            name = labels[id_]
            color1 = (0, 255, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color1, stroke, cv2.LINE_AA)
        img_captured_from_video = "lastframe.png"
        cv2.imwrite(img_captured_from_video, part_gray)

        # focusing face with a rectangle
        color = (255, 0, 0)
        color_stroke = 1
        end_coordinate_x = x + w
        end_coordinate_y = y + w
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y), color, color_stroke)

    # displaying the frames
    cv2.imshow("frame", frame)

    # exiting from window
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
