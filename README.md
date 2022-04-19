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