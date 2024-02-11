import os
import cv2
from labels_dict import labels_dict

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
words = []

while True:
    user_input = input("Enter the labels you want to create: (-1 to stop): ")
    if user_input == "-1":
        break
    words.append(user_input)

labels_dict = {word: word for word in words}

number_of_classes = len(words)
dataset_size = 300

cap = cv2.VideoCapture(0)
for j in range(len(words)):
    if not os.path.exists(os.path.join(DATA_DIR, words[j])):
        os.makedirs(os.path.join(DATA_DIR, words[j]))

    print("Collecting data for " + words[j])

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            'Ready? Press "Q" ! :)',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, words[j], "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
