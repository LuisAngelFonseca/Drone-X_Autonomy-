from djitellopy import Tello
from collections import deque
import cv2
import numpy as np
import time
import argparse
import imutils

global dirr

def display_grid(frame):
    """Display grid with 9 sectors on the video"""
    # Change the the frame of BGR to Gray
    # Display each line of the greed
    cv2.line(frame, pt1=(320, 0), pt2=(320, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(640, 0), pt2=(640, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, 240), pt2=(960, 240), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, 480), pt2=(960, 480), color=(255, 0, 0), thickness=2)
    return frame

def display_text(img_equ):
    """Display text in the video"""
    # Diplay text in the image
    font = cv2.FONT_ITALIC

    cv2.putText(img_equ, text='Drone-X', org=(410, 25), fontFace=font, fontScale=1, color=(255, 255, 255),
                thickness=2, lineType=cv2.LINE_8)

    return img_equ

def display_battery(img_equ):
    """Display a battery in the video that indicate the percentage of battery"""
    # Display a battery in the image
    cv2.rectangle(img_equ, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness= 2)
    cv2.rectangle(img_equ, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)
    battery = tello.get_battery() #Get battery level of the drone

    # Display a complete battery
    if battery > 75:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(938, 21), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(img_equ, pt1=(940, 9), pt2=(947, 21), color=(255, 255, 255), thickness=-1)
    #Display a 2/3 of the battery
    elif battery < 75 and battery > 50:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(940, 21), color=(255, 255, 255), thickness=-1)
    #Display 1/3 of the battery
    elif battery < 50 and battery > 25:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(255, 255, 255), thickness=-1)

    return img_equ

def procesing(frame):
    """Track the color in video"""
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Lemmons
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    pts = deque(maxlen=args["buffer"])

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        x = 480
        y = 360
        radius = 30

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    return(x, y, radius, frame)


def track_drone_track(x, y):
    if x < 640 and x > 320:
        if y < 480 and y > 240:
            dirr = 0
        elif y > 480:
            dirr = 4
        elif y < 240:
            dirr = 3
    elif y < 480 and y > 240:
        if x < 640 and x > 320:
            dirr = 0
        elif x > 640:
            dirr = 2
        elif x < 320:
            dirr = 1

    elif x < 320 and y < 240:
        dirr = 5
    elif x > 640 and y < 240:
        dirr = 6
    elif x < 320 and y > 480:
        dirr = 7
    elif x > 640 and y > 640:
        dirr = 8

    else:
        dirr = 0

    return dirr

def susana_distancia(r):
    if r > 20 and r < 40:
        mov = 0
    elif r > 40:
        mov = 1
    elif r < 20:
        mov = 2
    else:
        mov = 0
    return mov




tello = Tello()
tello.connect()
tello.streamon()
# Size of the image (960, 720)
counter = 0
dir = 0
time.sleep(2.0)
left_right_velocity = 0
for_back_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

while True:

    frame_read = tello.get_frame_read()
    frame = np.array(frame_read.frame)
    x, y, r, video = procesing(frame)
    dir = track_drone_track(x, y)
    mov = susana_distancia(r)
    video_2 = display_grid(video)
    video_user = display_battery(display_text(video_2))


    print(f"x = {x} y = {y} r = {r}")
    print(f"dir = {dir} mov = {mov} counter = {counter}")

    if counter == 60:
        tello.takeoff()


    if dir == 1:
        yaw_velocity = -25
    elif dir == 2:
        yaw_velocity = 25
    elif dir == 3:
        up_down_velocity = 25
    elif dir == 4:
        up_down_velocity = -25
    elif dir == 5:
        yaw_velocity = -25
        up_down_velocity = 25
    elif dir == 6:
        yaw_velocity = 25
        up_down_velocity = 25
    elif dir == 7:
        yaw_velocity = -25
        up_down_velocity = -25
    elif dir == 8:
         yaw_velocity = 25
         up_down_velocity = -25
    else:
        left_right_velocity = 0
        for_back_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0

    if dir == 0:
        if mov == 1:
            left_right_velocity = 0
            for_back_velocity = -20
            up_down_velocity = 0
            yaw_velocity = 0
        elif mov == 2:
            left_right_velocity = 0
            for_back_velocity = 20
            up_down_velocity = 0
            yaw_velocity = 0
        else:
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0




    if tello.send_rc_control:
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)


    counter = counter+1

    cv2.imshow('Drone X', video_user)
    if cv2.waitKey(20) & 0xFF == 'e':
        tello.emergency()

    if cv2.waitKey(20) & 0xFF == 27:
        break


# Never forget first to release
# And then to destroy
tello.stop_video_capture()
cap.release()
cv2.destroyAllWindows()






