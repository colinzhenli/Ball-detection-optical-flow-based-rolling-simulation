# Stage 1: Ball tracking

# from collections import deque
# from turtle import distance
# import numpy as np
# import argparse
# import imutils
# import cv2
# from matplotlib import pyplot as plt
#
# # def add_alpha_channel(img):
# #     # jpg图像添加alpha通道
# #     b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
# #     alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
# #     img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
# #     return img_new
#
#
# # the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
# MATCH_THRESHOLD = 0.93
# # the lower the threshold is, there must be more keypoints within the circle, the range is (0,1)
# WITHIN_CIRCLE_THRESHOLD = 0.43
# # The tolerant of the standard division, the bigger tolerant, the more candidate cycles
# stdTolerant = 90
# # The tolerant of the closest cycle's radius, used to define if the closest cycle is the ball,
# #  or random cycle on the floor
# rTolerant = 2
# # Previous center position of the best matching cycle
# old_pos = []
# # Previous radius of the best matching cycle
# old_r = 0
#
# frame_num = 0
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the (optional) video file")
# args = vars(ap.parse_args())
#
# greenLower = (0, 0, 0)
# greenUpper = (0, 0, 255)
# ball_img = cv2.imread('ball.png')
# if not args.get("video", False):
#     camera = cv2.VideoCapture(0)
# else:
#     camera = cv2.VideoCapture(args["video"])
# # fixed video path in my li's computer
# # camera = cv2.VideoCapture("/Users/lizhen/Desktop/CMPT_461（CV）/cmpt461/test.mp4")
#
# # Set the bool condition to initialize the video writer in the first frame
# initialized = False
# while True:
#     # read in the images and frames and change the format
#     (grabbed, frame) = camera.read()
#     if args.get("video") and not grabbed:
#         break
#     frame = imutils.resize(frame, width=600)
#     # 定义编解码器并创建VideoWriter对象
#     if not initialized:
#         width = frame.shape[1]
#         height = frame.shape[0]
#         size = (width, height)
#         print("width,height", size)
#         fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#         videoWriter = cv2.VideoWriter('out.avi', fourcc, 20.0, size)
#     initialized = True
#     # frame = cv2.resize(frame, (int(height*2), int(width*2)))
#     imageBallName = 'ball1.png'
#     imageBall = cv2.imread(imageBallName)
#     grayImageBall = cv2.cvtColor(imageBall, cv2.COLOR_BGR2GRAY)
#     grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect circles in the image
#     circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2=30, minRadius=5,
#                                maxRadius=300)
#     # circlesSTD = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
#     # creat SIFT and find the keypoints
#     sift = cv2.SIFT_create()
#     keypointBall, descriptorBall = sift.detectAndCompute(imageBall, None)
#     keypointFrame, descriptorFrame = sift.detectAndCompute(frame, None)
#
#     # find the matched keypoints by Brute-Force Matcher, the train image is the ball and the query image is the frame
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptorFrame, descriptorBall, k=2)
#     # Apply ratio test
#     goodMatches = []
#     for m, n in matches:
#         if m.distance < MATCH_THRESHOLD * n.distance:
#             goodMatches.append(m)
#     print([len(keypointBall), len(keypointFrame),
#            len(goodMatches)])  # print the number of keypoints and matched keypoints in frame
#
#     # get the list of matched points and list of their coordinates
#     matchedPointCoordinate = []
#     keypointFrameMatched = []
#     for matchedPoint in goodMatches:
#         keypointFrameMatched.append(keypointFrame[matchedPoint.queryIdx])
#         matchedPointCoordinate.append(keypointFrame[matchedPoint.queryIdx].pt)
#
#     # find the candidate circles containing many keypoints from the matched SIFT keypoints
#     minBlendDistance = np.inf
#     candidateCircle = []
#     minCircle = []
#     closestCircle = []
#     if (circles is not None) and (
#             len(goodMatches) != 0):  # if there circle number and matched points number are not zero
#         circle = np.round(circles[0, :]).astype(
#             "int")  # convert the (x, y) coordinates and radius of the circles to integers
#         # Initialize the minimum distance
#         if len(old_pos) > 0:
#             minDistance = np.inf
#         for (xCircle, yCircle, rCircle) in circle:  # loop over the (x, y) coordinates and radius of the circles
#             cv2.circle(frame, (xCircle, yCircle), rCircle, (0, 0, 0), 4)  # draw all the circles in black
#             totalDistance = 0
#             totalWithinCircle = 0
#             totalPoints = [(xCircle, yCircle)]
#             # Find the closest circle to the previous best matching circle
#             if len(old_pos) > 0:
#                 distance = np.sqrt(np.power(xCircle - old_pos[0], 2) + np.power(yCircle - old_pos[1], 2))
#                 if distance < minDistance:
#                     minDistance = distance
#                     closestCircle = [xCircle, yCircle, rCircle]
#                     print("Update closestCircle ", closestCircle)
#
#             for (xMatch, yMatch) in matchedPointCoordinate:  # loop over all the matched points
#                 currentDistance = np.sqrt(np.power(xCircle - xMatch, 2) + np.power(yCircle - yMatch, 2))
#                 totalDistance = totalDistance + currentDistance
#                 if currentDistance < rCircle:  # count the number of matched points within the circle
#                     totalWithinCircle += 1
#                     totalPoints.append((xMatch, yMatch))
#             # Compute the standard division
#             std = np.std(totalPoints)
#             if totalWithinCircle >= WITHIN_CIRCLE_THRESHOLD * len(
#                     goodMatches) and std <= stdTolerant and totalWithinCircle >= 4:
#                 candidateCircle.append((xCircle, yCircle, rCircle, totalDistance, std, totalWithinCircle))
#                 # draw all the candidate circles in white
#                 cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)
#     print([len(circle), len(candidateCircle)])  # print the number of circles and candidate circles
#
#     if len(candidateCircle) != 0:
#         # Fill the array by infinity
#         for index in range(3):
#             candidateCircle.append([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
#         # Sort the candidate circles by total distance
#         candidateCircleInDistance = sorted(candidateCircle, key=lambda x: x[3])
#         # Using the smallest circle that has standard division > stdTolerant
#         minCircle = (
#             (candidateCircleInDistance[0])[0], (candidateCircleInDistance[0])[1], (candidateCircleInDistance[0])[2])
#         print("特征点数", (candidateCircleInDistance[0])[5])
#         print("标准差 ", (candidateCircleInDistance[0])[4])
#
#     # draw the minimum circle in green if it exists
#     if len(minCircle) != 0:
#         cv2.circle(frame, (minCircle[0], minCircle[1]), minCircle[2], (0, 255, 0), 4)
#         old_pos = [minCircle[0], minCircle[1]]
#         old_r = minCircle[2]
#         print("old_pos ", old_pos)
#     # Using the closest circle as the best match if not found
#     elif len(closestCircle) != 0 and abs(old_r - closestCircle[2]) / closestCircle[2] <= rTolerant:
#         print("Using old_pos ", old_pos)
#         print("closest circle", closestCircle)
#         cv2.circle(frame, (closestCircle[0], closestCircle[1]), closestCircle[2], (0, 255, 0), 4)
#         old_pos = [closestCircle[0], closestCircle[1]]
#         old_r = closestCircle[2]
#     else:
#         # clear the old position, there is no ball in the scene, will not use closest cycle in this scene
#         old_pos = []
#
#     cv2.drawKeypoints(frame, keypointFrameMatched, frame, color=(255, 0, 255))  # draw matched keypoints in red
#     # frame = cv2.flip(frame, 0)  # 写翻转的框架
#     print("width", camera.get(3))  # 宽度
#     print("height", camera.get(4))  # 高度
#     videoWriter.write(frame)
#     cv2.imshow("Frame", frame)
#
#     frame_num = frame_num + 1
#
#     # 连续播放， 按‘q’退出程序
#     key = cv2.waitKey(1) & 0xFF
#     # 按任意键逐帧播放, 按‘q’退出程序
#     # key = cv2.waitKey(0) & 0xFF
#     if key == ord("q"):
#         break
#
# camera.release()
# videoWriter.release()
# cv2.destroyAllWindows()




from collections import deque
from turtle import distance, st
import numpy as np
import argparse
import scipy
import imutils
import matlab.engine
import cv2
from matplotlib import pyplot as plt
from scipy import signal
import seaborn as sns

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
# the multipler of the flow speed
MULTIPLER = 1.3
# the window size in optical flow
WINDOWSIZE = 25

class Frame:
    def __init__(self, frameNum, frame, center, radius):
        self.frameNum = frameNum
        self.frame = frame
        self.circle = self.Circle(center, radius)
    class Circle:
        def __init__(self, center, radius):
            self.center = center
            self.radius = radius

#create mask and call blend function from matlab
def blendFromMatlab(frame, croppedFrame):
    x = ((frame.circle).center)[0]
    y = ((frame.circle).center)[1]
    r = (frame.circle).radius
    target  = frame.frame
    mask = np.zeros((target.shape[0], target.shape[1]))
    source = np.zeros((target.shape))
    source[y-r:y+r, x-r:x+r] = croppedFrame

    #get the mask
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            if(((r-j)**2 + (r-i)**2) <= r**2):
                mask[i, j] = 1
    eng = matlab.engine.start_matlab()
    blended = eng.imblend(source, mask, target, 0)
    return blended

# resize the mapping to the object size
def mappingResize(mapping, objectHeight, objectWidth):
    newMapping = np.zeros((objectHeight, objectWidth, 2))
    sourceHeight = (mapping.shape)[0]
    sourceWidth = (mapping.shape)[1]
    for i in range(objectHeight):
        for j in range(objectWidth):
            indexI = np.round(i*sourceHeight/objectHeight)
            indexJ = np.round(j*sourceWidth/objectWidth)
            indexI = indexI.astype("int")
            indexJ = indexJ.astype("int")
            newMapping[i, j] = mapping[indexI, indexJ]
    return newMapping

#decide whether the circle is totally in the frame
def circleInRange(frame, circle):
    if ((circle[0] - circle[2]) < 0):
        return False
    if ((circle[0] + circle[2]) > frame.shape[1]):
        return False
    if ((circle[1] - circle[2]) < 0):
        return False
    if ((circle[1] - circle[2]) > frame.shape[0]):
        return False
    return True
#replace the circle with the texture
def BlendFrame(frame, texture, mapping):
    #get the center and radius of two circles
    x = ((frame.circle).center)[0]
    y = ((frame.circle).center)[1]
    r = (frame.circle).radius
    #crop the two image frame into two square region
    croppedFrame = frame.frame[y-r:y+r, x-r:x+r].copy()
    for i in range(2*r-1):
        for j in range(2*r-1):
            # update the circle
            if(((r-i)**2 + (r-j)**2) <= r**2):
                croppedFrame[i, j] = texture[(mapping[i, j, 0]%(texture.shape[0])-1), (mapping[i, j, 1]%(texture.shape[1])-1)]

    return croppedFrame

#update the mapping on the circle
def UpdateMap(opticalFlow, mapping):
    height = (np.shape(opticalFlow))[0]
    width = (np.shape(opticalFlow))[0]
    mapping = mapping + opticalFlow
    mapping = mapping.astype("int")
    return mapping

#compute the optical flow within a circle between two frames
def OpticalFlow(frame1, frame2):
    #get the center and radius of two circles
    x1 = ((frame1.circle).center)[0]
    x2 = ((frame2.circle).center)[0]
    y1 = ((frame1.circle).center)[1]
    y2 = ((frame2.circle).center)[1]
    r1 = (frame1.circle).radius
    r2 = (frame2.circle).radius

    #crop the two image frame into two square region
    croppedFrame1 = frame1.frame[y1-r1:y1+r1, x1-r1:x1+r1]
    croppedFrame2 = frame2.frame[y2-r2:y2+r2, x2-r2:x2+r2]
    #resize the source frame
    croppedFrame1 = cv2.resize(croppedFrame1, (croppedFrame2.shape[0], croppedFrame2.shape[1]))
    sourceFrame = cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2GRAY)
    targetFrame = cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2GRAY)
    targetFrame = targetFrame / 255
    sourceFrame = sourceFrame / 255
    windowSize = WINDOWSIZE
    threshold = 1e-2
    xKernel = np.array([[-1., 1.], [-1., 1.]])
    yKernel = np.array([[-1., -1.], [1., 1.]])
    tKernel = np.array([[1., 1.], [1., 1.]])
    mode = 'same'
    dx = signal.convolve2d(sourceFrame, xKernel, boundary='symm', mode=mode)
    dy = signal.convolve2d(sourceFrame, yKernel, boundary='symm', mode=mode)
    dt = signal.convolve2d(targetFrame, tKernel, boundary='symm', mode=mode) + signal.convolve2d(sourceFrame, -tKernel, boundary='symm', mode=mode)
    u = np.zeros((2*r2, 2*r2))
    v = np.zeros((2*r2, 2*r2))
    flow = np.zeros((2*r2, 2*r2, 2))
    # within window window_size * window_size
    for i in range(targetFrame.shape[0]):
        for j in range(targetFrame.shape[1]):
            Ix = []
            Iy = []
            It = []
            # for pixels within the circle
            if (((r2 - i) ** 2 + (r2 - j) ** 2) <= r2 ** 2):
                # for the pixels within the window and circle
                for indexI in range(i - windowSize, i + windowSize):
                    for indexJ in range(j - windowSize, j + windowSize):
                        if (((r2 - indexI) ** 2 + (r2 - indexJ) ** 2) < r2 ** 2):
                            Ix.append(dx[indexI, indexJ])
                            Iy.append(dy[indexI, indexJ])
                            It.append(dt[indexI, indexJ])
                # Ix = dx[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
                # Iy = dy[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
                # It = dt[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
                Ix = np.array(Ix)
                Iy = np.array(Iy)
                It = np.array(It)
                b = np.reshape(It, (It.shape[0], 1))
                A = np.vstack((Ix, Iy)).T
                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= threshold:
                    nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]
                    flow[i, j] = (nu[0], nu[1])
    flow = np.floor(flow*MULTIPLER)
    flow = flow.astype("int")
    return (flow)

frameSet = []
frame_num = 0
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

greenLower = (0, 0, 0)
greenUpper = (0, 0, 255)
ball_img = cv2.imread('ball.png')
ball2_img = cv2.imread('ball2.png')
texture_img = cv2.imread('texture2.png')
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
    # if args.get("video") and not grabbed:
    if not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    # 定义编解码器并创建VideoWriter对象
    if not initialized:
        width = frame.shape[1]
        height = frame.shape[0]
        size = (width, height)
        print("width,height", size)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        videoWriter = cv2.VideoWriter('soccerout.avi', fourcc, 20.0, size)
    initialized = True
    # frame = cv2.resize(frame, (int(height*2), int(width*2)))
    imageBallName = 'ball.png'
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
            # cv2.circle(frame, (xCircle, yCircle), rCircle, (0, 0, 0), 4)  # draw all the circles in black
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
                # cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)
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
        # cv2.circle(frame, (minCircle[0], minCircle[1]), minCircle[2], (0, 255, 0), 4)
        old_pos = [minCircle[0], minCircle[1]]
        old_r = minCircle[2]
        print("old_pos ", old_pos)
        #add the frame with circle to frame set every frame
        if ((frame_num!=0) and circleInRange(frame, minCircle)):
            circleFrame = Frame(frame_num, frame, (minCircle[0], minCircle[1]), minCircle[2])
            if (len(frameSet)!=0):
                lastFrame = frameSet[-1]
                flow = OpticalFlow(lastFrame, circleFrame)
                mapping = mappingResize(mapping, flow.shape[0], flow.shape[1])
                mapping = UpdateMap(flow, mapping)
                croppedFrame = BlendFrame(circleFrame, texture_img, mapping)
                frame = blendFromMatlab(circleFrame, croppedFrame)
            else:
                mapping = np.zeros((2 * minCircle[2], 2 * minCircle[2], 2))
                startPointX = np.floor((texture_img.shape[0] - mapping.shape[0]) / 2)
                startPointY = np.floor((texture_img.shape[1] - mapping.shape[1]) / 2)
                startPoint = np.array([startPointX, startPointY])
                startPoint = startPoint.astype("int")
                for i in range(mapping.shape[0]):
                    for j in range(mapping.shape[1]):
                        #initial the mapping to the center of texture image
                        mapping[i, j, 0] = i + startPoint[0]
                        mapping[i, j, 1] = j + startPoint[1]
            frameSet.append(circleFrame)
    # Using the closest circle as the best match if not found
    elif len(closestCircle) != 0 and abs(old_r - closestCircle[2]) / closestCircle[2] <= rTolerant:
        print("Using old_pos ", old_pos)
        print("closest circle", closestCircle)
        # cv2.circle(frame, (closestCircle[0], closestCircle[1]), closestCircle[2], (0, 255, 0), 4)
        old_pos = [closestCircle[0], closestCircle[1]]
        old_r = closestCircle[2]
        #add the frame with circle to frame set every frame
        if ((frame_num!=0) and circleInRange(frame, closestCircle)):
            circleFrame = Frame(frame_num, frame, (closestCircle[0], closestCircle[1]), closestCircle[2])
            if (len(frameSet)!=0):
                lastFrame = frameSet[-1]
                flow = OpticalFlow(lastFrame, circleFrame)
                mapping = mappingResize(mapping, flow.shape[0], flow.shape[1])
                mapping = UpdateMap(flow, mapping)
                croppedFrame = BlendFrame(circleFrame, texture_img, mapping)
                frame = blendFromMatlab(circleFrame, croppedFrame)
            else:
                mapping = np.zeros((2 * closestCircle[2], 2 * closestCircle[2], 2))
                for i in range(mapping.shape[0]):
                    for j in range(mapping.shape[1]):
                        #initial the mapping to the center of texture image
                        mapping[i, j, 0] = i + startPoint[0]
                        mapping[i, j, 1] = j + startPoint[1]
            frameSet.append(circleFrame)
    else:
        # clear the old position, there is no ball in the scene, will not use closest cycle in this scene
        old_pos = []

    # cv2.drawKeypoints(frame, keypointFrameMatched, frame, color=(255, 0, 255))  # draw matched keypoints in red
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






# from collections import deque
# from turtle import distance, st
# import numpy as np
# import argparse
# import scipy
# import imutils
# # import matlab.engine
# import cv2
# from matplotlib import pyplot as plt
# from scipy import signal
# import seaborn as sns
#
# # the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
# MATCH_THRESHOLD = 0.9
# # the lower the threshold is, there must be more keypoints within the circle, the range is (0,1)
# WITHIN_CIRCLE_THRESHOLD = 0.6
# # The tolerant of the standard division, the bigger tolerant, the more candidate cycles
# stdTolerant = 90
# # The tolerant of the closest cycle's radius, used to define if the closest cycle is the ball,
# #  or random cycle on the floor
# rTolerant = 1.5
# # Previous center position of the best matching cycle
# old_pos = []
# # Previous radius of the best matching cycle
# old_r = 0
# # the multipler of the flow speed
# MULTIPLER = 20
# # the window size in optical flow
# WINDOWSIZE = 15
#
# class Frame:
#     def __init__(self, frameNum, frame, center, radius):
#         self.frameNum = frameNum
#         self.frame = frame
#         self.circle = self.Circle(center, radius)
#     class Circle:
#         def __init__(self, center, radius):
#             self.center = center
#             self.radius = radius
#
# #create mask and call blend function from matlab
# # def blendFromMatlab(frame, croppedFrame):
# #     x = ((frame.circle).center)[0]
# #     y = ((frame.circle).center)[1]
# #     r = (frame.circle).radius
# #     target  = frame.frame
# #     # mask = np.zeros((target.shape[0], target.shape[1]))
# #     # source = np.zeros((target.shape))
# #     # source[y-r:y+r, x-r:x+r] = croppedFrame
# #     source = croppedFrame
# #     mask = np.zeros(source.shape, source.dtype)
#
# #     #get the mask
# #     for i in range(source.shape[0]):
# #         for j in range(source.shape[1]):
# #             # if(((r-j)**2 + (r-i)**2) <= r**2):
# #             mask[i, j] = [1, 1, 1]
# #     # eng = matlab.engine.start_matlab()
# #     # blended = eng.imblend(source, mask, target, 0)
# #     # cv2.imshow("Frame", frame.frame)
# #     output = cv2.seamlessClone(source, target, mask, (y, x), cv2.MIXED_CLONE)
# #     return output
#
# # resize the mapping to the object size
# def mappingResize(mapping, objectHeight, objectWidth):
#     newMapping = np.zeros((objectHeight, objectWidth, 2))
#     sourceHeight = (mapping.shape)[0]
#     sourceWidth = (mapping.shape)[1]
#     for i in range(objectHeight):
#         for j in range(objectWidth):
#             indexI = np.round(i*sourceHeight/objectHeight)
#             indexJ = np.round(j*sourceWidth/objectWidth)
#             indexI = indexI.astype("int")
#             indexJ = indexJ.astype("int")
#             newMapping[i, j] = mapping[indexI, indexJ]
#     return newMapping
#
# #decide whether the circle is totally in the frame
# def circleInRange(frame, circle):
#     if ((circle[0] - circle[2]) < 0):
#         return False
#     if ((circle[0] + circle[2]) > frame.shape[1]):
#         return False
#     if ((circle[1] - circle[2]) < 0):
#         return False
#     if ((circle[1] - circle[2]) > frame.shape[0]):
#         return False
#     return True
# #replace the circle with the texture
# def BlendFrame(frame, texture, mapping):
#     #get the center and radius of two circles
#     x = ((frame.circle).center)[0]
#     y = ((frame.circle).center)[1]
#     r = (frame.circle).radius
#     #crop the two image frame into two square region
#     croppedFrame = frame.frame[y-r:y+r, x-r:x+r].copy()
#     for i in range(2*r-1):
#         for j in range(2*r-1):
#             # update the circle
#             if(((r-i)**2 + (r-j)**2) <= r**2):
#                 croppedFrame[i, j] = texture[(mapping[i, j, 0]%(texture.shape[0])-1), (mapping[i, j, 1]%(texture.shape[1])-1)]
#
#     frame.frame[y-r:y+r, x-r:x+r] = croppedFrame
#     return frame.frame
#
# #update the mapping on the circle
# def UpdateMap(opticalFlow, mapping):
#     height = (np.shape(opticalFlow))[0]
#     width = (np.shape(opticalFlow))[0]
#     mapping = mapping + opticalFlow
#     mapping = mapping.astype("int")
#     return mapping
#
# #compute the optical flow within a circle between two frames
# def OpticalFlow(frame1, frame2):
#     #get the center and radius of two circles
#     x1 = ((frame1.circle).center)[0]
#     x2 = ((frame2.circle).center)[0]
#     y1 = ((frame1.circle).center)[1]
#     y2 = ((frame2.circle).center)[1]
#     r1 = (frame1.circle).radius
#     r2 = (frame2.circle).radius
#
#     #crop the two image frame into two square region
#     croppedFrame1 = frame1.frame[y1-r1:y1+r1, x1-r1:x1+r1]
#     croppedFrame2 = frame2.frame[y2-r2:y2+r2, x2-r2:x2+r2]
#     #resize the source frame
#     croppedFrame1 = cv2.resize(croppedFrame1, (croppedFrame2.shape[0], croppedFrame2.shape[1]))
#     sourceFrame = cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2GRAY)
#     targetFrame = cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2GRAY)
#     targetFrame = targetFrame / 255
#     sourceFrame = sourceFrame / 255
#     windowSize = WINDOWSIZE
#     threshold = 1e-2
#     xKernel = np.array([[-1., 1.], [-1., 1.]])
#     yKernel = np.array([[-1., -1.], [1., 1.]])
#     tKernel = np.array([[1., 1.], [1., 1.]])
#     mode = 'same'
#     dx = signal.convolve2d(sourceFrame, xKernel, boundary='symm', mode=mode)
#     dy = signal.convolve2d(sourceFrame, yKernel, boundary='symm', mode=mode)
#     dt = signal.convolve2d(targetFrame, tKernel, boundary='symm', mode=mode) + signal.convolve2d(sourceFrame, -tKernel, boundary='symm', mode=mode)
#     u = np.zeros((2*r2, 2*r2))
#     v = np.zeros((2*r2, 2*r2))
#     flow = np.zeros((2*r2, 2*r2, 2))
#     # within window window_size * window_size
#     for i in range(windowSize, targetFrame.shape[0]-windowSize):
#         for j in range(windowSize, targetFrame.shape[1]-windowSize):
#             Ix = []
#             Iy = []
#             It = []
#             # for pixels within the circle
#             if(((r2-i)**2 + (r2-j)**2) <= r2**2):
#                 # for the pixels within the window and circle
#                 for indexI in range(i-windowSize, i+windowSize):
#                     for indexJ in range(j-windowSize, j+windowSize):
#                         if(((r2-indexI)**2 + (r2-indexJ)**2) <= r2**2):
#                             Ix.append(dx[indexI, indexJ])
#                             Iy.append(dy[indexI, indexJ])
#                             It.append(dt[indexI, indexJ])
#                 # Ix = dx[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
#                 # Iy = dy[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
#                 # It = dt[i-windowSize:i+windowSize, j-windowSize:j+windowSize].flatten()
#                 Ix = np.array(Ix)
#                 Iy = np.array(Iy)
#                 It = np.array(It)
#                 b = np.reshape(It, (It.shape[0],1))
#                 A = np.vstack((Ix, Iy)).T
#                 if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= threshold:
#                     nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
#                     u[i,j]=nu[0]
#                     v[i,j]=nu[1]
#                     flow[i, j] = (nu[0], nu[1])
#     flow = np.floor(flow*MULTIPLER)
#     flow = flow.astype("int")
#     return (flow)
#
# frameSet = []
# frame_num = 0
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the (optional) video file")
# args = vars(ap.parse_args())
#
# greenLower = (0, 0, 0)
# greenUpper = (0, 0, 255)
# ball_img = cv2.imread('ball2.png')
# ball2_img = cv2.imread('ball2.png')
# texture_img = cv2.imread('texture.png')
# if not args.get("video", False):
#     camera = cv2.VideoCapture(0)
# else:
#     camera = cv2.VideoCapture(args["video"])
# # fixed video path in my li's computer
# # camera = cv2.VideoCapture("/Users/lizhen/Desktop/CMPT_461（CV）/cmpt461/test2.mp4")
#
# # Set the bool condition to initialize the video writer in the first frame
# initialized = False
# while True:
#     # read in the images and frames and change the format
#     (grabbed, frame) = camera.read()
#     # if args.get("video") and not grabbed:
#     if not grabbed:
#         break
#     frame = imutils.resize(frame, width=600)
#     # 定义编解码器并创建VideoWriter对象
#     if not initialized:
#         width = frame.shape[1]
#         height = frame.shape[0]
#         size = (width, height)
#         print("width,height", size)
#         fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#         videoWriter = cv2.VideoWriter('out.avi', fourcc, 20.0, size)
#     initialized = True
#     # frame = cv2.resize(frame, (int(height*2), int(width*2)))
#     imageBallName = 'basketball.png'
#     imageBall = cv2.imread(imageBallName)
#     grayImageBall = cv2.cvtColor(imageBall, cv2.COLOR_BGR2GRAY)
#     grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect circles in the image
#     circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2=30, minRadius=5,
#                                maxRadius=300)
#     # circlesSTD = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
#     # creat SIFT and find the keypoints
#     sift = cv2.SIFT_create()
#     keypointBall, descriptorBall = sift.detectAndCompute(imageBall, None)
#     keypointFrame, descriptorFrame = sift.detectAndCompute(frame, None)
#
#     # find the matched keypoints by Brute-Force Matcher, the train image is the ball and the query image is the frame
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptorFrame, descriptorBall, k=2)
#     # Apply ratio test
#     goodMatches = []
#     for m, n in matches:
#         if m.distance < MATCH_THRESHOLD * n.distance:
#             goodMatches.append(m)
#     print([len(keypointBall), len(keypointFrame),
#            len(goodMatches)])  # print the number of keypoints and matched keypoints in frame
#
#     # get the list of matched points and list of their coordinates
#     matchedPointCoordinate = []
#     keypointFrameMatched = []
#     for matchedPoint in goodMatches:
#         keypointFrameMatched.append(keypointFrame[matchedPoint.queryIdx])
#         matchedPointCoordinate.append(keypointFrame[matchedPoint.queryIdx].pt)
#
#     # find the candidate circles containing many keypoints from the matched SIFT keypoints
#     minBlendDistance = np.inf
#     candidateCircle = []
#     minCircle = []
#     closestCircle = []
#     if (circles is not None) and (
#             len(goodMatches) != 0):  # if there circle number and matched points number are not zero
#         circle = np.round(circles[0, :]).astype(
#             "int")  # convert the (x, y) coordinates and radius of the circles to integers
#         # Initialize the minimum distance
#         if len(old_pos) > 0:
#             minDistance = np.inf
#         for (xCircle, yCircle, rCircle) in circle:  # loop over the (x, y) coordinates and radius of the circles
#             # cv2.circle(frame, (xCircle, yCircle), rCircle, (0, 0, 0), 4)  # draw all the circles in black
#             totalDistance = 0
#             totalWithinCircle = 0
#             totalPoints = [(xCircle, yCircle)]
#             # Find the closest circle to the previous best matching circle
#             if len(old_pos) > 0:
#                 distance = np.sqrt(np.power(xCircle - old_pos[0], 2) + np.power(yCircle - old_pos[1], 2))
#                 if distance < minDistance:
#                     minDistance = distance
#                     closestCircle = [xCircle, yCircle, rCircle]
#                     print("Update closestCircle ", closestCircle)
#
#             for (xMatch, yMatch) in matchedPointCoordinate:  # loop over all the matched points
#                 currentDistance = np.sqrt(np.power(xCircle - xMatch, 2) + np.power(yCircle - yMatch, 2))
#                 totalDistance = totalDistance + currentDistance
#                 if currentDistance < rCircle:  # count the number of matched points within the circle
#                     totalWithinCircle += 1
#                     totalPoints.append((xMatch, yMatch))
#             # Compute the standard division
#             std = np.std(totalPoints)
#             if totalWithinCircle >= WITHIN_CIRCLE_THRESHOLD * len(
#                     goodMatches) and std <= stdTolerant and totalWithinCircle >= 4:
#                 candidateCircle.append((xCircle, yCircle, rCircle, totalDistance, std, totalWithinCircle))
#                 # draw all the candidate circles in white
#                 # cv2.circle(frame, (xCircle, yCircle), rCircle, (255, 255, 255), 4)
#     print([len(circle), len(candidateCircle)])  # print the number of circles and candidate circles
#
#     if len(candidateCircle) != 0:
#         # Fill the array by infinity
#         for index in range(3):
#             candidateCircle.append([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
#         # Sort the candidate circles by total distance
#         # candidateCircleInDistance = sorted(candidateCircle, key=lambda x: x[3])
#         candidateCircleInDistance = sorted(candidateCircle, key=lambda x: x[5])
#         # Using the smallest circle that has standard division > stdTolerant
#         minCircle = (
#             (candidateCircleInDistance[0])[0], (candidateCircleInDistance[0])[1], (candidateCircleInDistance[0])[2])
#         print("特征点数", (candidateCircleInDistance[0])[5])
#         print("标准差 ", (candidateCircleInDistance[0])[4])
#
#     # draw the minimum circle in green if it exists
#     if len(minCircle) != 0 and len(old_pos) == 0:
#         cv2.circle(frame, (minCircle[0], minCircle[1]), minCircle[2], (0, 255, 0), 4)
#         old_pos = [minCircle[0], minCircle[1]]
#         old_r = minCircle[2]
#         print("old_pos ", old_pos)
#         # #add the frame with circle to frame set every frame
#         # if ((frame_num!=0) and circleInRange(frame, minCircle)):
#         #     circleFrame = Frame(frame_num, frame, (minCircle[0], minCircle[1]), minCircle[2])
#         #     if (len(frameSet)!=0):
#         #         lastFrame = frameSet[-1]
#         #         flow = OpticalFlow(lastFrame, circleFrame)
#         #         mapping = mappingResize(mapping, flow.shape[0], flow.shape[1])
#         #         mapping = UpdateMap(flow, mapping)
#         #         frame = BlendFrame(circleFrame, texture_img, mapping)
#         #         # croppedFrame = BlendFrame(circleFrame, texture_img, mapping)
#         #         # frame = blendFromMatlab(circleFrame, croppedFrame)
#         #     else:
#         #         mapping = np.zeros((2 * minCircle[2], 2 * minCircle[2], 2))
#         #         startPointX = np.floor((texture_img.shape[0] - mapping.shape[0]) / 2)
#         #         startPointY = np.floor((texture_img.shape[1] - mapping.shape[1]) / 2)
#         #         startPoint = np.array([startPointX, startPointY])
#         #         startPoint = startPoint.astype("int")
#         #         for i in range(mapping.shape[0]):
#         #             for j in range(mapping.shape[1]):
#         #                 #initial the mapping to the center of texture image
#         #                 mapping[i, j, 0] = i + startPoint[0]
#         #                 mapping[i, j, 1] = j + startPoint[1]
#         #     frameSet.append(circleFrame)
#     # Using the closest circle as the best match if not found
#     # elif len(closestCircle) != 0 and abs(old_r - closestCircle[2]) / closestCircle[2] <= rTolerant:
#     #     print("Using old_pos ", old_pos)
#     #     print("closest circle", closestCircle)
#     #     cv2.circle(frame, (closestCircle[0], closestCircle[1]), closestCircle[2], (0, 255, 0), 4)
#     #     old_pos = [closestCircle[0], closestCircle[1]]
#     #     old_r = closestCircle[2]
#     #     #add the frame with circle to frame set every frame
#     #     # if ((frame_num!=0) and circleInRange(frame, closestCircle)):
#     #     #     circleFrame = Frame(frame_num, frame, (closestCircle[0], closestCircle[1]), closestCircle[2])
#     #     #     if (len(frameSet)!=0):
#     #     #         lastFrame = frameSet[-1]
#     #     #         flow = OpticalFlow(lastFrame, circleFrame)
#     #     #         mapping = mappingResize(mapping, flow.shape[0], flow.shape[1])
#     #     #         mapping = UpdateMap(flow, mapping)
#     #     #         frame = BlendFrame(circleFrame, texture_img, mapping)
#     #     #         # croppedFrame = BlendFrame(circleFrame, texture_img, mapping)
#     #     #         # frame = blendFromMatlab(circleFrame, croppedFrame)
#     #     #     else:
#     #     #         mapping = np.zeros((2 * closestCircle[2], 2 * closestCircle[2], 2))
#     #     #         for i in range(mapping.shape[0]):
#     #     #             for j in range(mapping.shape[1]):
#     #     #                 #initial the mapping to the center of texture image
#     #     #                 mapping[i, j, 0] = i + startPoint[0]
#     #     #                 mapping[i, j, 1] = j + startPoint[1]
#     #     #     frameSet.append(circleFrame)
#     elif len(old_pos) > 0:
#         # clear the old position, there is no ball in the scene, will not use closest cycle in this scene
#         # old_pos = []
#         print("Using old_pos ", old_pos)
#         cv2.circle(frame, (old_pos[0], old_pos[1]), old_r, (0, 255, 0), 4)
#         # add the frame with circle to frame set every frame
#         # if ((frame_num!=0) and circleInRange(frame, closestCircle)):
#         #     circleFrame = Frame(frame_num, frame, (closestCircle[0], closestCircle[1]), closestCircle[2])
#         #     if (len(frameSet)!=0):
#         #         lastFrame = frameSet[-1]
#         #         flow = OpticalFlow(lastFrame, circleFrame)
#         #         mapping = mappingResize(mapping, flow.shape[0], flow.shape[1])
#         #         mapping = UpdateMap(flow, mapping)
#         #         frame = BlendFrame(circleFrame, texture_img, mapping)
#         #         # croppedFrame = BlendFrame(circleFrame, texture_img, mapping)
#         #         # frame = blendFromMatlab(circleFrame, croppedFrame)
#         #     else:
#         #         mapping = np.zeros((2 * closestCircle[2], 2 * closestCircle[2], 2))
#         #         for i in range(mapping.shape[0]):
#         #             for j in range(mapping.shape[1]):
#         #                 #initial the mapping to the center of texture image
#         #                 mapping[i, j, 0] = i + startPoint[0]
#         #                 mapping[i, j, 1] = j + startPoint[1]
#         #     frameSet.append(circleFrame)
#
#     cv2.drawKeypoints(frame, keypointFrameMatched, frame, color=(255, 0, 255))  # draw matched keypoints in red
#     # frame = cv2.flip(frame, 0)  # 写翻转的框架
#     print("width", camera.get(3))  # 宽度
#     print("height", camera.get(4))  # 高度
#     videoWriter.write(frame)
#     cv2.imshow("Frame", frame)
#
#     frame_num = frame_num + 1
#
#     # 连续播放， 按‘q’退出程序
#     key = cv2.waitKey(1) & 0xFF
#     # 按任意键逐帧播放, 按‘q’退出程序
#     # key = cv2.waitKey(0) & 0xFF
#     if key == ord("q"):
#         break
#
# camera.release()
# videoWriter.release()
# cv2.destroyAllWindows()


