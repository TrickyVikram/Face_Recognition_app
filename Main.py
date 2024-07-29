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
