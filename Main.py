import os
import cv2
import cvzone
import numpy as np
import face_recognition
import pickle
from PIL import Image

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from  dotenv import load_dotenv

# Specify the path to your Firebase service account JSON file
json_file_path = "./ServiceAccountKey.json"
databaseURL=os.getenv('databaseURL')

try:
    # Attempt to open the JSON file
    with open(json_file_path) as f:
        cred = credentials.Certificate(json_file_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL
        })

    # Reference to the database path where you want to insert data
    ReferencePath = db.reference('Students')
except FileNotFoundError:
    print(f"File not found: {json_file_path}. Please check the file path and try again.")




# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) # set width
cap.set(4, 480) # set height

# Load the background image
img_background = cv2.imread("./Resources/Background.png")

# Load the images from the "Modes" folder
folderModesPath = "./Resources/Modes/"
modepathList = os.listdir(folderModesPath)
imgModesList = []

# Read each image and append it to the list
for path in modepathList:
    img = cv2.imread(f"{folderModesPath}{path}")
    imgModesList.append(img)



modeType = 4

# Resize the mode image to be used
imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))

#encdoe load file 


try:
    print("Loading encoded file data...")
    with open("encoded_file.p", "rb") as file:
        loaded_data = pickle.load(file)
        file.close()
        encodeListKnown, StudentsId = loaded_data["encodings"], loaded_data["student_ids"]
        print(StudentsId)
        print("Loading encoded file data...")
        # print("Loaded data:", loaded_data)

except Exception as e:
    print(f"Error loading encoded data: {e}")

counter=0


while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image from webcam.")
        continue

    imgS= cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Compare the faces in the current frame with the known faces
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)
        matchIndex = np.argmin(faceDis)
        match_percentage = (1 - faceDis[matchIndex]) * 100
        print("match index", matchIndex)
        Students_Id = StudentsId[matchIndex]

        if matches[matchIndex] and match_percentage >= 5:
           Students_Id = StudentsId[matchIndex]
           match_value = match_percentage
           y1, x2, y2, x1 = faceLoc
           y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

           # Using cvzone to draw the rectangle and put text
           cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), rt=0)  # Draws a rectangle with rounded corners
        #    cvzone.putTextRect(img, f"{match_value:.2f}%", (x1 + 6, y2 - 10), scale=1, thickness=2, offset=10)
        
           if counter ==0:
              counter = 1

    if counter !=0:

        if counter==1:
            studentInfo=db.reference('Students/'+str(Students_Id)).get()
            print(studentInfo)
        
            # Data to be inserted
        cvzone.putTextRect(img, f"Name: {studentInfo['name']}, Roll: {studentInfo['roll']}, Batch: {studentInfo['Batch']}", (x1 + 6, y2 - 10), scale=1, thickness=2, offset=10)

        
        counter+=1

           
 
    # Display the webcam feed with the detected faces
    cv2.imshow("Webcam", img_background)

    # Place the webcam feed on the background
    img_background[376:376+480, 220:220+640] = img

    # Place the resized mode image on the background
    img_background[152:152+798, 1238:1238+482] = imgModesList_Shape

    # Display the result
    cv2.imshow("Face Attendance", img_background)



     # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
       break


# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
