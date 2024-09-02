import cv2
import numpy as np
import handTrack as ht
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 15
#########################

print(1)
prevTime=0
curTime=0
cam=cv2.VideoCapture(0)
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cam.set(3, wCam)
cam.set(4, hCam)
detector = ht.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1. Finding the hand Landmarks
    success, img = cam.read()
    img = detector.findHands(img)
    lmList ,bbox= detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        #print(lmList)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)
        if fingers[1]==1 and fingers[2]==0:
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            #to reduce the shaking of the mouse cursor
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img, (x1, y1), 15, (255,255,0),cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    
    #11. Frame
    curTime=time.time()
    fps=1/(curTime-prevTime)
    prevTime=curTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,0),2)

    #12. display
    cv2.imshow("Image", img)
    cv2.waitKey(1)