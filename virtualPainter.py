import cv2
import numpy as np
import time
import HandTrackingModule as htm
import os



folderPath = "header"
brushThikness = 10
eraserThickness = 100
mylist = os.listdir(folderPath)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (172,56,48)
blue = (172,56,48)
green = (37,173,98)
yellow = (26,191,209)


detector = htm.handDetector(False,1,0.85,0.85)
xp,yp = 0,0


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

image_canvas = np.zeros((720,1280,3),np.uint8)
while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    
    lmlist = detector.findPosition(img,draw = False)
    
    if len(lmlist)!=0:
        
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        
        fingers = detector.fingersUp()
        
        
        if fingers[1] == 0 and fingers[2] == 0 :
            xp,yp = 0,0
            
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
            if y1<125:
                if 10<x1<150:
                    drawColor = blue
                elif 200<x1<350:
                    
                    drawColor = green
                elif 400<x1<600:
                   
                    drawColor = yellow
                elif 600<x1<892:
                    drawColor = (0,0,0)
            
            
        if fingers[1] == 0 and fingers[2] == 1 :
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp = x1,y1
            
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(image_canvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            
            cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThikness)
             
            cv2.line(image_canvas,(xp,yp),(x1,y1),drawColor,brushThikness)
            xp,yp = x1,y1
            
            
    imgGray = cv2.cvtColor(image_canvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,image_canvas)
    
    
    img[0:87,0:892] = header
    img = cv2.addWeighted(img,0.5,image_canvas,0.5,0)
    cv2.imshow("Image",img)
    
    k = cv2.waitKey(1)
    if k == 13:
        break
    
