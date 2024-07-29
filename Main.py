import cv2
import os

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

# Resize the mode image to be used
imgModesList_Shape = cv2.resize(imgModesList[0], (482, 798))

while True:
    success, img = cap.read()
    


    # Detect faces in the image
    Faces_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = Faces_detect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces and add labels
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text_y = y - 10 if y - 10 > 10 else y + h + 20 
        # cv2.putText(img, "Face", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the webcam feed with the detected faces
    cv2.imshow("Webcam", img)





   
    
    # Place the webcam feed on the background
    img_background[376:376+480, 220:220+640] = img

    # Place the resized mode image on the background
    img_background[152:152+798, 1238:1238+482] = imgModesList_Shape

    # Display the result
    cv2.imshow("Face Attendance", img_background)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
