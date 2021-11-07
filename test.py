from typing import FrozenSet
from tracker import Motion, select_rectangle
import cv2
import time

cam=cv2.VideoCapture("reddot.mp4")

rectangle=select_rectangle(cam, 100) #returns with x,y h, w
print(rectangle)

m=Motion(camera=cam)
m.track(100,200, rectangle, "CSRT")
print(m.rectangle_path)
print(len(m.rectangle_path))

cam.set(cv2.CAP_PROP_POS_FRAMES,100)
for i in range(len(m.rectangle_path)):
    x,y,w,h=m.rectangle_path[i]
    ret, frame=cam.read()
    frame=cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
    cv2.imshow("frame", frame)
    time.sleep(0.05)
    print(cam.get(cv2.CAP_PROP_POS_FRAMES))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()