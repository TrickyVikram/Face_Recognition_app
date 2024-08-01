import cv2
import os
import face_recognition
import pickle

# Load the students' images from the images folder
folderPath = "Images"
if not os.path.exists(folderPath):
    print(f"Directory {folderPath} does not exist.")

pathList = os.listdir(folderPath)
imgList = []
StudentsId = []

# Read each image and append it to the list
for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)

    # Get the student's id
    StudentsId.append(os.path.splitext(path)[0])

print(StudentsId)

# Encode the images

# Encode the images
print("Encoding started")
def encodeImages(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return

# Encode the images
print("Start Complete")
encodeListKnown = encodeImages(imgList)
print("Encoding Complete")
print(encodeListKnown)
