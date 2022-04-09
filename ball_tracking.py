from collections import deque
from turtle import distance
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

# the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
MATCH_THRESHOLD = 0.93
# the higher the threshold is, there must be more keypoints within the circle, the range is (0,1)
WITHIN_CIRCLE_THRESHOLD = 0.45
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
    if (circles is not None) and (len(goodMatches) != 0):  # if there circle number and matched points number are not zero
        circle = np.round(circles[0, :]).astype("int")  # convert the (x, y) coordinates and radius of the circles to integers

        for (xCircle, yCircle, rCircle) in circle:  # loop over the (x, y) coordinates and radius of the circles
            cv2.circle(frame, (xCircle, yCircle), rCircle, (0, 0, 0), 4)  # draw all the circles in black
            totalDistance = 0
            totalWithinCircle = 0
            for (xMatch, yMatch) in matchedPointCoordinate:  # loop over all the matched points
                currentDistance = np.sqrt(np.power(xCircle - xMatch, 2) + np.power(yCircle - yMatch, 2))
                totalDistance = totalDistance + currentDistance
                if (currentDistance < rCircle):  # count the number of matched points within the circle
                    totalWithinCircle += 1
            if ((totalWithinCircle) >= WITHIN_CIRCLE_THRESHOLD * len(goodMatches)):
                candidateCircle.append((xCircle, yCircle, rCircle, totalDistance))
                cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)  # draw all the circles in white
    print([len(circle), len(candidateCircle)])  # print the number of circles and candidate circles

    # sort the candidate circles by the total distances from all matched keypoints to the centre of the circle
    # the order is ascending
    # find the smallest circle within the first 2 candidates
    if (len(candidateCircle) != 0):
        candidateCircle = sorted(candidateCircle, key=lambda x: x[3])
        rMin = (candidateCircle[0])[2]
        for index in range(3):
            candidateCircle.append([np.inf, np.inf, np.inf, np.inf])
        for index in range(2):
            if (candidateCircle[index])[2] <= rMin:
                minCircle = ((candidateCircle[index])[0], (candidateCircle[index])[1], (candidateCircle[index])[2])

    # draw the minimum circle in green if it exists
    if (len(minCircle) != 0):
        cv2.circle(frame, (minCircle[0], minCircle[1]), minCircle[2], (0, 255, 0), 4)
    cv2.drawKeypoints(frame, keypointFrameMatched, frame, color=(255, 0, 255))  # draw matched keypoints in red
    cv2.imshow("Frame", frame)

    # 按任意键逐帧播放, 按‘q’退出程序
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
