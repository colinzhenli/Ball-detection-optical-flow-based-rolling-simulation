# cmpt461

python ball_tracking.py --video test.mp4

# test:
# the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
MATCH_THRESHOLD = 0.93
# the lower the threshold is, there must be more keypoints within the circle, the range is (0,1)
WITHIN_CIRCLE_THRESHOLD = 0.43
# The tolerant of the standard division, the bigger tolerant, the more candidate cycles
stdTolerant = 90
# The tolerant of the closest cycle's radius, used to define if the closest cycle is the ball,
#  or random cycle on the floor
rTolerant = 2
# detect circles in the image
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2=30, minRadius=5,
                               maxRadius=300)

#test3:
python ball_tracking.py --video test3.mp4
# the higher the threshold is, the more matched keypoints in frame, the range is (0,1)
MATCH_THRESHOLD = 0.8
# the lower the threshold is, there must be more keypoints within the circle, the range is (0,1)
WITHIN_CIRCLE_THRESHOLD = 0.3
# The tolerant of the standard division, the bigger tolerant, the more candidate cycles
stdTolerant = 160
# The tolerant of the closest cycle's radius, used to define if the closest cycle is the ball,
#  or random cycle on the floor
rTolerant = 2