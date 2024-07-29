import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)#width
cap.set(4, 480)#height

img_background = cv2.imread("./Resources/Background.png")

# Load the images list from the "Modes" folder
folderModesPath = "./Resources/Modes/"
modepathList = os.listdir(folderModesPath)
imgModesList = []

# Read each image and append it to the list
for path in modepathList:
    imgModesList.append(cv2.imread(f"{folderModesPath}{path}"))




while True:
    sucess, img = cap.read()
 
 

    img_background[376:376+480, 220:220+640]=img

    img_background[10:10+50, 500:500+50]=imgModesList[1]
   


    cv2.imshow("Face Attendance", img_background)
    cv2.waitKey(1)

   
    
