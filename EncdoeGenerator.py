import os
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


# Specify the path to your Firebase service account JSON file
json_file_path = "./ServiceAccountKey.json"
databaseURL='https://collegeattendence-fff29-default-rtdb.firebaseio.com/'

# Firebase Initialization
try:
    with open(json_file_path) as f:
        cred = credentials.Certificate(json_file_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL,
            'storageBucket': 'collegeattendence-fff29.appspot.com'
        })

except FileNotFoundError:
 print(f"File not found: {json_file_path}. Please check the file path and try again.")
 exit()

# Load the students' images from the images folder
folderPath = "Images"
if not os.path.exists(folderPath):
    print(f"Directory {folderPath} does not exist.")
    exit()

pathList = os.listdir(folderPath)
imgList = []
StudentsId = []

# Read each image and append it to the list
for path in pathList:
    if path == ".DS_Store":
        continue  # Skip .DS_Store file
    
    img_path = os.path.join(folderPath, path)
    try:
        img = Image.open(img_path)
        img = img.convert("RGB")
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgList.append(img_np)
        StudentsId.append(os.path.splitext(path)[0])

        # Save the image to the storage bucket
        fileName = img_path
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)
        print(f"Image {fileName} uploaded successfully to Firebase storage.")
        
    except Exception as e:
        print(f"Error processing image {path}: {e}")

print("Student IDs:", StudentsId)

# Function to encode the images
def encodeImages(images):
    encodedList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encodedList.append(encodings[0])
            else:
                print("No face found in one of the images.")
        except Exception as e:
            print(f"Error encoding image: {e}")
    return encodedList

# Encode the images
print("Encoding started")
encodeListKnown = encodeImages(imgList)
print("Encoding Complete")

# Check if encoding was successful
if encodeListKnown:
    # Save the encoded list and student IDs to a file
    data = {"encodings": encodeListKnown, "student_ids": StudentsId}
    with open("encoded_file.p", "wb") as f:
        pickle.dump(data, f)
    print("Encodings and IDs saved to encoded_file.p")
else:
    print("No encodings were generated.")

# Optional: Load and check the saved data
try:
    with open("encoded_file.p", "rb") as f:
        loaded_data = pickle.load(f)
        print("Loaded data:", loaded_data)
except Exception as e:
    print(f"Error loading encoded data: {e}")

# End of file
