from collections import deque
from turtle import distance
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt


# def add_alpha_channel(img):
#     # jpg图像添加alpha通道
#     b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
#     alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
#     img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
#     return img_new


# the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
MATCH_THRESHOLD = 0.93
# the lower the threshold is, there must be more keypoints within the circle, the range is (0,1)
WITHIN_CIRCLE_THRESHOLD = 0.43
# The tolerant of the standard division, the bigger tolerant, the more candidate cycles
stdTolerant = 90
# The tolerant of the closest cycle's radius, used to define if the closest cycle is the ball,
#  or random cycle on the floor
rTolerant = 2
# Previous center position of the best matching cycle
old_pos = []
# Previous radius of the best matching cycle
old_r = 0

frame_num = 0
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

greenLower = (0, 0, 0)
greenUpper = (0, 0, 255)
ball_img = cv2.imread('ball.png')
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
# fixed video path in my li's computer
# camera = cv2.VideoCapture("/Users/lizhen/Desktop/CMPT_461（CV）/cmpt461/test.mp4")

# 定义编解码器并创建VideoWriter对象
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = camera.get(4)
# height = camera.get(3)
size = (width, height)
print("width,height", size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
videoWriter = cv2.VideoWriter('out.avi', fourcc, 20.0, size)

while True:
    # read in the images and frames and change the format
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    imageBallName = 'test2.png'
    imageBall = cv2.imread(imageBallName)
    grayImageBall = cv2.cvtColor(imageBall, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.7, 100, param1=100, param2=30, minRadius=5,
                               maxRadius=300)
    # circlesSTD = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
    # creat SIFT and find the keypoints
    sift = cv2.SIFT_create()
    keypointBall, descriptorBall = sift.detectAndCompute(imageBall, None)
    keypointFrame, descriptorFrame = sift.detectAndCompute(frame, None)

    # find the matched keypoints by Brute-Force Matcher, the train image is the ball and the query image is the frame
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorFrame, descriptorBall, k=2)
    # Apply ratio test
    goodMatches = []
    for m, n in matches:
        if m.distance < MATCH_THRESHOLD * n.distance:
            goodMatches.append(m)
    print([len(keypointBall), len(keypointFrame),
           len(goodMatches)])  # print the number of keypoints and matched keypoints in frame

    # get the list of matched points and list of their coordinates
    matchedPointCoordinate = []
    keypointFrameMatched = []
    for matchedPoint in goodMatches:
        keypointFrameMatched.append(keypointFrame[matchedPoint.queryIdx])
        matchedPointCoordinate.append(keypointFrame[matchedPoint.queryIdx].pt)

    # find the candidate circles containing many keypoints from the matched SIFT keypoints
    minBlendDistance = np.inf
    candidateCircle = []
    minCircle = []
    closestCircle = []
    if (circles is not None) and (
            len(goodMatches) != 0):  # if there circle number and matched points number are not zero
        circle = np.round(circles[0, :]).astype(
            "int")  # convert the (x, y) coordinates and radius of the circles to integers
        # Initialize the minimum distance
        if len(old_pos) > 0:
            minDistance = np.inf
        for (xCircle, yCircle, rCircle) in circle:  # loop over the (x, y) coordinates and radius of the circles
            cv2.circle(frame, (xCircle, yCircle), rCircle, (0, 0, 0), 4)  # draw all the circles in black
            totalDistance = 0
            totalWithinCircle = 0
            totalPoints = [(xCircle, yCircle)]
            # Find the closest circle to the previous best matching circle
            if len(old_pos) > 0:
                distance = np.sqrt(np.power(xCircle - old_pos[0], 2) + np.power(yCircle - old_pos[1], 2))
                if distance < minDistance:
                    minDistance = distance
                    closestCircle = [xCircle, yCircle, rCircle]
                    print("Update closestCircle ", closestCircle)

            for (xMatch, yMatch) in matchedPointCoordinate:  # loop over all the matched points
                currentDistance = np.sqrt(np.power(xCircle - xMatch, 2) + np.power(yCircle - yMatch, 2))
                totalDistance = totalDistance + currentDistance
                if currentDistance < rCircle:  # count the number of matched points within the circle
                    totalWithinCircle += 1
                    totalPoints.append((xMatch, yMatch))
            # Compute the standard division
            std = np.std(totalPoints)
            if totalWithinCircle >= WITHIN_CIRCLE_THRESHOLD * len(goodMatches) and std <= stdTolerant:
                candidateCircle.append((xCircle, yCircle, rCircle, totalDistance, std, totalWithinCircle))
                # draw all the candidate circles in white
                cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)
    print([len(circle), len(candidateCircle)])  # print the number of circles and candidate circles

    # sort the candidate circles by the total distances from all matched keypoints to the centre of the circle
    # the order is ascending
    # find the smallest circle within the first 2 candidates
    # if (len(candidateCircle) != 0):
    #     candidateCircle = sorted(candidateCircle, key=lambda x: x[3])
    #     rMin = (candidateCircle[0])[2]
    #     for index in range(3):
    #         candidateCircle.append([np.inf, np.inf, np.inf, np.inf])
    #     for index in range(2):
    #         if (candidateCircle[index])[2] <= rMin:
    #             minCircle = ((candidateCircle[index])[0], (candidateCircle[index])[1], (candidateCircle[index])[2])
    #             print("totalDistance", (candidateCircle[index])[3])
    #             print("标准差 ", (candidateCircle[index])[4])

    # sort the candidate circles by the total distances from all matched keypoints to the centre of the circle
    # the order is ascending
    # find the smallest circle within the first 2 candidates

    if len(old_pos) > 0:
        src1 = cv2.imread('C:\PyCharm_Code\cmpt461\mask.png', cv2.IMREAD_COLOR)
        mask = cv2.resize(src1, (2 * old_r, 2 * old_r), interpolation=cv2.INTER_NEAREST)
        xTopLeft = old_pos[0] - old_r
        yTopLeft = old_pos[1] + old_r
        xBottomRight = old_pos[0] + old_r
        yBottomRight = old_pos[1] - old_r
        # cv2.imwrite('C:\PyCharm_Code\cmpt461\currentFrame' + '.png', frame)
        # src2 = cv2.imread('C:\PyCharm_Code\cmpt461\currentFrame.png', cv2.IMREAD_COLOR)
        # frame = src2
        # frame = merge_img(src2, mask, yTopLeft, yBottomRight, xTopLeft, xBottomRight)
        # add_alpha_channel(frame)
        if 0 < xTopLeft < frame.shape[1] and \
                0 < xBottomRight < frame.shape[1] and \
                0 < yTopLeft < frame.shape[0] and \
                0 < yBottomRight < frame.shape[0]:
            frame[yBottomRight:yTopLeft, xTopLeft:xBottomRight] = mask

    if len(candidateCircle) != 0:
        # Fill the array by infinity
        for index in range(3):
            candidateCircle.append([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # Sort the candidate circles by total distance
        candidateCircleInDistance = sorted(candidateCircle, key=lambda x: x[3])
        # Using the smallest circle that has standard division > stdTolerant
        minCircle = ((candidateCircleInDistance[0])[0], (candidateCircleInDistance[0])[1], (candidateCircleInDistance[0])[2])
        print("特征点数", (candidateCircleInDistance[0])[5])
        print("标准差 ", (candidateCircleInDistance[0])[4])

    # draw the minimum circle in green if it exists
    if len(minCircle) != 0:
        cv2.circle(frame, (minCircle[0], minCircle[1]), minCircle[2], (0, 255, 0), 4)
        old_pos = [minCircle[0], minCircle[1]]
        old_r = minCircle[2]
        print("old_pos ", old_pos)
    # Using the closest circle as the best match if not found
    elif len(closestCircle) != 0 and abs(old_r - closestCircle[2]) / closestCircle[2] <= rTolerant:
        print("Using old_pos ", old_pos)
        print("closest circle", closestCircle)
        cv2.circle(frame, (closestCircle[0], closestCircle[1]), closestCircle[2], (0, 255, 0), 4)
        old_pos = [closestCircle[0], closestCircle[1]]
        old_r = closestCircle[2]
    else:
        # clear the old position, there is no ball in the scene, will not use closest cycle in this scene
        old_pos = []

    cv2.drawKeypoints(frame, keypointFrameMatched, frame, color=(255, 0, 255))  # draw matched keypoints in red
    # frame = cv2.flip(frame, 0)  # 写翻转的框架
    print("width", camera.get(3))  # 宽度
    print("height", camera.get(4))  # 高度
    videoWriter.write(frame)
    cv2.imshow("Frame", frame)

    frame_num = frame_num + 1

    # 连续播放， 按‘q’退出程序
    key = cv2.waitKey(1) & 0xFF
    # 按任意键逐帧播放, 按‘q’退出程序
    # key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

camera.release()
videoWriter.release()
cv2.destroyAllWindows()
