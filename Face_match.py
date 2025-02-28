import os
import cv2
import numpy as np
import face_recognition
import pickle
import time
import csv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Firebase Configuration
json_file_path = "./ServiceAccountKey.json"
databaseURL = os.getenv('databaseURL')
storageBucket = os.getenv('storageBucket')

# Firebase Initialization
try:
    with open(json_file_path) as f:
        cred = credentials.Certificate(json_file_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL,
            'storageBucket': storageBucket
        })
except FileNotFoundError:
    print(f"File not found: {json_file_path}. Please check the file path and try again.")
    exit()

# Path to the attendance CSV file
attendance_file = 'attendance.csv'

# Ensure the CSV file exists and has headers
if not os.path.isfile(attendance_file):
    with open(attendance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student_ID', 'last_attendance'])  # Write header if file is new

# Function to mark attendance in Firebase Database
def mark_attendance(student_id):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Get current time in a readable format

    # Get a reference to the Students node in Firebase Realtime Database
    ref = db.reference('Students')

    # Check if this student exists in the database
    attendance_ref = ref.order_by_child('Student_ID').equal_to(student_id).limit_to_last(1).get()

    if attendance_ref:
        # If student exists, check if the 'last_attendance' field exists
        for key, value in attendance_ref.items():
            last_marked_time = value.get('last_attendance', None)  # Use .get() to avoid KeyError
            
            # If 'last_attendance' doesn't exist, treat it as the first attendance
            if last_marked_time:
                # Convert the string 'last_marked_time' to timestamp
                last_marked_timestamp = time.mktime(time.strptime(last_marked_time, "%Y-%m-%d %H:%M:%S"))
                
                # Get current timestamp
                current_timestamp = time.time()

                # Check if attendance was marked within the last 20 seconds
                if current_timestamp - last_marked_timestamp < 20:
                    print(f"Attendance already marked for {student_id} within 20 seconds.")
                    return
            else:
                print(f"No previous attendance found for {student_id}. Marking first attendance.")
    else:
        print(f"Student {student_id} not found in database. Adding to database.")
    
    # Save or update the attendance data in Firebase
    ref.push({
        'Student_ID': student_id,
        'last_attendance': current_time,
    })

    print(f"Attendance marked for {student_id} at {current_time}")

    # Log attendance to CSV file
    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_id, current_time])  # Append the student's attendance to CSV

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set width
cap.set(4, 480)  # set height

# Load the background image
img_background = cv2.imread("./Resources/Background.png")

# Load the images from the "Modes" folder
folderModesPath = "./Resources/Modes/"
modepathList = os.listdir(folderModesPath)
imgModesList = []

# Read each image and append it to the list
for path in modepathList:
    img = cv2.imread(f"{folderModesPath}{path}")
    if img is None:
        print(f"Warning: Failed to load image at {folderModesPath}{path}")
    else:
        imgModesList.append(img)

# Resize the mode images
imgModesList_Shape = [cv2.resize(img, (482, 798)) for img in imgModesList if img is not None]

# Load encoded data
try:
    print("Loading encoded file data...")
    with open("encoded_file.p", "rb") as file:
        loaded_data = pickle.load(file)
        encodeListKnown, StudentsId = loaded_data["encodings"], loaded_data["student_ids"]
        print("Encoded data loaded successfully.")
except Exception as e:
    print(f"Error loading encoded data: {e}")
    exit()

modeType = 0  # Default mode type if no match or face
last_marked_time = {}  # Dictionary to store last marked time for each student

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image from webcam.")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    Students_Id = ''
    match_value = None

    if len(faceCurFrame) > 0:  # If faces are detected
        # Compare the faces in the current frame with the known faces
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            matchIndex = np.argmin(faceDis)
            match_percentage = (1 - faceDis[matchIndex]) * 100
            
            if matches[matchIndex] and match_percentage >= 50:  # Match
                Students_Id = StudentsId[matchIndex].upper()
                match_value = match_percentage
                modeType = 2  # Set mode to 5 on successful match
                current_time = time.time()  # Get current timestamp

                # Check if this student has already been marked within the last 10 seconds
                if Students_Id in last_marked_time:
                    time_diff = current_time - last_marked_time[Students_Id]
                    if time_diff < 10:  # If less than 10 seconds
                        modeType = 4  # Set to mode 6 for "Already Marked"
                    else:
                        last_marked_time[Students_Id] = current_time  # Update the time if it's more than 10 seconds
                        mark_attendance(Students_Id)  # Mark attendance in Firebase
                else:
                    last_marked_time[Students_Id] = current_time  # First time marking attendance
                    mark_attendance(Students_Id)  # Mark attendance in Firebase
                
                # Display face match percentage and bounding box for known faces
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{match_value:.2f}%", (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                # Display bounding box for unknown faces
                modeType = 4  # Set to default mode for unknown faces
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for unknown faces
                cv2.putText(img, "Unknown", (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        modeType = 1  # Set to default mode if no faces detected

    # Place the webcam feed on the background
    img_background[376:376 + 480, 220:220 + 640] = img

    # Place the corresponding mode image on the background
    if modeType < len(imgModesList_Shape):
        img_background[152:152 + 798, 1238:1238 + 482] = imgModesList_Shape[modeType]

    # Display the student ID on the screen
    cv2.putText(img_background, f"Student ID :", (220, 850), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img_background, f" {Students_Id}", (405, 850), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Face Attendance", img_background)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
