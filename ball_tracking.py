# Stage 1: Ball tracking

from collections import deque
from turtle import distance
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

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

# Set the bool condition to initialize the video writer in the first frame
initialized = False
while True:
    # read in the images and frames and change the format
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    # 定义编解码器并创建VideoWriter对象
    if not initialized:
        width = frame.shape[1]
        height = frame.shape[0]
        size = (width, height)
        print("width,height", size)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        videoWriter = cv2.VideoWriter('out.avi', fourcc, 20.0, size)
    initialized = True
    # frame = cv2.resize(frame, (int(height*2), int(width*2)))
    imageBallName = 'ball1.png'
    imageBall = cv2.imread(imageBallName)
    grayImageBall = cv2.cvtColor(imageBall, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2=30, minRadius=5,
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
            if totalWithinCircle >= WITHIN_CIRCLE_THRESHOLD * len(
                    goodMatches) and std <= stdTolerant and totalWithinCircle >= 4:
                candidateCircle.append((xCircle, yCircle, rCircle, totalDistance, std, totalWithinCircle))
                # draw all the candidate circles in white
                cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)
    print([len(circle), len(candidateCircle)])  # print the number of circles and candidate circles

    if len(candidateCircle) != 0:
        # Fill the array by infinity
        for index in range(3):
            candidateCircle.append([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # Sort the candidate circles by total distance
        candidateCircleInDistance = sorted(candidateCircle, key=lambda x: x[3])
        # Using the smallest circle that has standard division > stdTolerant
        minCircle = (
            (candidateCircleInDistance[0])[0], (candidateCircleInDistance[0])[1], (candidateCircleInDistance[0])[2])
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
