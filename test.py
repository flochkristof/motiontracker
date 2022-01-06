from typing import FrozenSet
from tracker import Motion, select_rectangle, select_point
import cv2
import time

cam = cv2.VideoCapture("reddot.mp4")

rectangle = select_rectangle(cam, 1000)  # returns with x,y h, w
cv2.destroyAllWindows()
print(rectangle)

m = Motion(camera=cam)
m.track(100, 400, rectangle, "CSRT")
print(m.timestamp)
print(m.rectangle_path)
print(m.status)

cam.set(cv2.CAP_PROP_POS_FRAMES, 100)
for i in range(len(m.rectangle_path)):
    x, y, w, h = m.rectangle_path[i]
    ret, frame = cam.read()
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("frame", frame)
    print(cam.get(cv2.CAP_PROP_POS_FRAMES))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
