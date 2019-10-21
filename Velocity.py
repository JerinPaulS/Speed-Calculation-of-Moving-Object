import cv2
import imutils
import math
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

"""
850 pixels in my frame equals 4 meter in the real world
"""

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

video = cv2.VideoCapture('/home/jerinpaul/Downloads/Ball1.mp4')
frame_rate = video.get(5)
cnt = 0
sec_cnt = 0
first_frame = None
last_frame = None
second_frame = None
flag = 0
distance = 0
t1 = 3
t2 = 4
time = t2 - t1

if video.isOpened()== False:
    print("Error opening video stream or file")

while video.isOpened():
    ret, frame = video.read()
    frame_id = video.get(1)
    if ret == True:

        frame = imutils.resize(frame, width=1000)
        frame = frame[150 : 250, 100 : 950]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cnt == 0:
            first_frame = gray
            second_frame = gray
            last_frame = gray
            cnt = cnt + 1

        frameDelta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frameDelta, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        if frame_id % math.floor(frame_rate) == 0:
            sec_cnt = sec_cnt + 1
            if sec_cnt == t1:
                second_frame = thresh
            if sec_cnt == t2:
                last_frame = thresh
                flag = 1

        if flag == 1:
            last_frame = cv2.add(second_frame, last_frame)
            gray = cv2.GaussianBlur(last_frame, (7, 7), 0)
            edged = cv2.Canny(gray, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
            	if cv2.contourArea(c) < 100:
            		continue
            	box = cv2.minAreaRect(c)
            	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            	box = np.array(box, dtype="int")
            	box = perspective.order_points(box)
            	cX = np.average(box[:, 0])
            	cY = np.average(box[:, 1])
                (tl, tr, br, bl) = box
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                distance = D

            #cv2.imshow('SuperImpose', edged)
            #cv2.waitKey(25)

        #cv2.imshow('Frame',thresh)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()

distance = distance / 39.37
speed = distance / time
print(str(speed) + "m/sec")
