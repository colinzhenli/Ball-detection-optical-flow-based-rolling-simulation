from collections import deque
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
# Resources：
# OpenCV中文使用手册： http://woshicver.com/ThirdSection/2_1_%E5%9B%BE%E5%83%8F%E5%85%A5%E9%97%A8/


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
ball_img = cv2.imread('ball.png')
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

    kernel_size = (1, 1)
    sigma = cv2.BORDER_DEFAULT


    # img = cv2.GaussianBlur(img, kernel_size, sigma)

    imgname1 = 'C:/PyCharm_Code/cmpt461/test2.png'
    # imgname2 = 'C:/PyCharm_Code/cmpt461/test1.png'
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
    img1 = cv2.imread(imgname1)
    # img1 = cv2.GaussianBlur(img1, kernel_size, sigma)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    img2 = frame
    # img2 = cv2.GaussianBlur(img2, kernel_size, sigma)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    # hmerge = np.hstack((gray1, gray2))  # 水平拼接
    # cv2.imshow("gray", hmerge)  # 拼接显示为gray
    # cv2.waitKey(0)

    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    # sift.detectAndComputer(gray， None)计算出图像的关键点和sift特征向量   参数说明：gray表示输入的图片
    # des1表示sift特征向量，128维
    print("图片1的关键点数目：" + str(len(kp1)))
    # print(des1.shape)
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子
    print("图片2的关键点数目：" + str(len(kp2)))

    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    # img3 = cv2.drawKeypoints(gray, kp, img) 在图中画出关键点   参数说明：gray表示输入图片, kp表示关键点，img表示输出的图片
    # print(img3.size)
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

    # hmerge = np.hstack((img3, img4))  # 水平拼接
    # cv2.imshow("point", img4)  # 拼接显示为gray
    # cv2.waitKey(0)

    # # BFMatcher解决匹配
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    # # print(matches)
    # # print(len(matches))
    # # 调整ratio
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])
    # # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    # img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.imshow("BFmatch", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






    cv2.imshow("Frame", frame)
    #按任意键逐帧播放
    # key = cv2.waitKey(0) & 0xFF
    # # 按‘q’退出程序
    # if key == ord("q"):
    #     break

    # 按‘q’退出程序
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

# # 如果能获取摄像头则打开摄像头， 否则打开视频
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
