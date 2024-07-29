import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)#width
cap.set(4, 480)#height

img_background = cv2.imread("./Resources/Background.png")

while True:
    sucess, img = cap.read()
 
 
  

    img_background[376:376+480, 220:220+640]=img
    cv2.imshow("Face Attendance", img_background)
    cv2.waitKey(1)

   
