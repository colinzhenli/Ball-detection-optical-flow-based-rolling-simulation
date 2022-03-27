from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# Resources：
# OpenCV中文使用手册： http://woshicver.com/ThirdSection/2_1_%E5%9B%BE%E5%83%8F%E5%85%A5%E9%97%A8/


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# 如果能获取摄像头则打开摄像头， 否则打开视频
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    # camera.read()按帧读取视频
    # 如果读取帧是正确的grabbed = True, frame就是每一帧的图像，是个三维矩阵
    (grabbed, frame) = camera.read()

    # 如果视频读取完就退出程序
    if args.get("video") and not grabbed:
        break
    cv2.imshow("Frame", frame)


    #按任意键逐帧播放
    key = cv2.waitKey(0) & 0xFF
    # 按‘q’退出程序
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

# 如果能获取摄像头则打开摄像头， 否则打开视频
# if not args.get("video", False):
#     camera = cv2.VideoCapture(0)
# else:
#     camera = cv2.VideoCapture(args["video"])
#
# while True:
#     # camera.read()按帧读取视频
#     # 如果读取帧是正确的grabbed = True, frame就是每一帧的图像，是个三维矩阵
#     (grabbed, frame) = camera.read()
#
#     if args.get("video") and not grabbed:
#         break
#
#     frame = imutils.resize(frame, width=600)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # cv2.imshow("Frame", frame)
#
#     mask = cv2.inRange(hsv, greenLower, greenUpper)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)
#
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     center = None
#
#     if len(cnts) > 0:
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#
#         if radius > 10:
#             cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
#
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", mask)
#
#     #按‘q’退出程序
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
#
# camera.release()
# cv2.destroyAllWindows()
