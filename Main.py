
"""
This script performs face recognition and attendance marking using a webcam.
The script performs the following steps:
1. Imports necessary libraries and modules.
2. Loads environment variables from a .env file.
3. Initializes Firebase using the service account JSON file.
4. Initializes the webcam and sets its width and height.
5. Loads the background image.
6. Loads the images from the "Modes" folder.
7. Loads the encoded file data.
8. Enters an infinite loop to continuously capture frames from the webcam.
9. Resizes and converts the captured frame to RGB.
10. Detects faces and encodes them in the current frame.
11. Compares the detected faces with known faces.
12. If a match is found and the match percentage is above a threshold, marks attendance and displays student information.
13. If no face is detected, resets the mode and counter.
14. Displays the webcam feed and mode image on the background.
15. Breaks the loop and exits the script on 'q' key press.
16. Releases the webcam and closes windows.
Note: Make sure to provide the correct file paths for the background image, mode images, encoded file, and Firebase service account JSON file.
"""
import os
import cv2
import cvzone
import numpy as np
import face_recognition
import pickle
from PIL import Image
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Specify the path to your Firebase service account JSON file
json_file_path = "./ServiceAccountKey.json"
databaseURL = os.getenv('databaseURL')
storageBucket = os.getenv('storageBucket')

try:
    # Initialize Firebase
    with open(json_file_path) as f:
        cred = credentials.Certificate(json_file_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL,
            'storageBucket': storageBucket
        }) 
except FileNotFoundError:
    print(f"File not found: {json_file_path}. Please check the file path and try again.")

bucket = storage.bucket()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

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




# Load encoded file
try:
    print("Loading encoded file data...")
    with open("encoded_file.p", "rb") as file:
        loaded_data = pickle.load(file)
        encodeListKnown, StudentsId = loaded_data["encodings"], loaded_data["student_ids"]
        print(StudentsId)
except Exception as e:
    print(f"Error loading encoded data: {e}")

modeType = 4
counter = 0
imgStudent = []


while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image from webcam.")
        continue

   

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all faces and face encodings in the current frame of video
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))

    if faceCurFrame:
    # Compare faces in the current frame with known faces
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            match_percentage = (1 - faceDis[matchIndex]) * 100
            Students_Id = StudentsId[matchIndex]

            if matches[matchIndex] and match_percentage >= 5:
                Students_Id = StudentsId[matchIndex]
                match_value = match_percentage
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), rt=0)

            if counter ==0:
                cvzone.putTextRect(img, "Loding", (275,400))
                cv2.imshow("Face Attendance", img_background)
                cv2.waitKey(10)
                counter = 1
                modeType = 4
                
                
                

        if counter != 0:
            if counter == 1:
            


            # Get student info from Firebase

                studentInfo = db.reference(f'Students/{Students_Id}').get()
                print(studentInfo)

                # Get image from Firebase
                blob = bucket.get_blob(f"Images/{Students_Id}.png")
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                Student_img = cv2.imdecode(array, cv2.COLOR_BGR2GRAY)

                # Update the student attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendence'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)


            if secondsElapsed < 10:
                # Update Firebase reference
                ReferencePath = db.reference(f'Students/{Students_Id}')

                # Assuming 'total_attendence' should be an integer count, initialize if not already integer
                try:
                    total_attendence = int(studentInfo['total_attendence'])
                    total_attendence += 1
                    
                    ReferencePath.child('total_attendence').set(total_attendence)
                    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ReferencePath.child('last_attendence').set(current_datetime_str)
                except ValueError:
                    print("Error:")
               
            else:
                print("Attendance not marked. Time elapsed is more than 1 Hours .")
                modeType = 1
                counter = 0
                imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))


    else:
        modeType = 4
        counter = 0
        imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))




        if modeType!=1:
            if 30 < counter <= 40:
                modeType = 2
                imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))
                if counter <= 30:
                    modeType = 3
                    cv2.putText(imgModesList_Shape, f"{studentInfo['roll']}", (165, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)
                    cv2.putText(imgModesList_Shape, f"{studentInfo['name']}", (200, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)
                    cv2.putText(imgModesList_Shape, f"{studentInfo['Batch']}", (207, 591), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

                    imgStudent = cv2.resize(Student_img, (216, 216))
                    imgModesList_Shape[185:185+216, 113:113+216] = imgStudent

            counter += 1

            if counter == 45:
                counter = 0
                modeType = 4
                studentInfo = []
                Student_img = []
                imgModesList_Shape = cv2.resize(imgModesList[modeType], (482, 798))
                
                


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