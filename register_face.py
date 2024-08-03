import cv2
import os

# Get the student ID as input
student_id = input("Enter student ID: ")

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Specify the folder where images should be saved
folderPath = "Images"

# Check if the folder exists; if not, create it
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
    print(f"Directory {folderPath} created.")

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start capturing video
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(200, 200))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = frame[y:y + h, x:x + w]

    # Show the frame with the detected face(s)
    cv2.imshow("Face", frame)

    # Wait for user input to save or quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save the image
        if len(faces) > 0:
            image_filename = f"{folderPath}/{student_id}.jpg"
            # image_fullImg = f"{folderPath}/{student_id}.jpg"
            cv2.imwrite(image_filename, face)
            # cv2.imwrite(image_fullImg, frame)
            print(f"Face image saved as {image_filename}")
        else:
            print("No face detected. Image not saved.")
        break
    elif key == ord('q'):  # Press 'q' to quit without saving
        print("Image not saved. Exiting.")
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
