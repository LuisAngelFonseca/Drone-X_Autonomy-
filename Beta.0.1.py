from djitellopy import Tello
from collections import deque
import cv2
import numpy as np
import time
import argparse
import imutils


def display_grid(frame):
    """Display grid on the video"""
    # Change the the frame of BGR to Gray
    # Display each line of the greed
    x1 = 460-100
    x2 = 500+100
    y1 = 340-65
    y2 = 380+65
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
    return x1, x2, y1, y2, frame

def display_text(img_equ):
    """Display text in the video"""
    # Diplay text in the image
    font = cv2.FONT_ITALIC

    cv2.putText(img_equ, text='Drone-X', org=(410, 25), fontFace=font, fontScale=1, color=(0, 0, 0),
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
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(938, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(img_equ, pt1=(940, 9), pt2=(947, 21), color=(0, 255, 0), thickness=-1)
    #Display a 2/3 of the battery
    elif battery < 75 and battery > 50:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 255), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(940, 21), color=(0, 255, 255), thickness=-1)
    #Display 1/3 of the battery
    elif battery < 50 and battery > 25:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)

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

    # Lemmons color range
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


def track_drone_track(x, y, limitx1, limitx2, limity1, limity2):
    dirr = 0
    if x < limitx2 and x > limitx1:
        if y < limity2 and y > limity1:
            dirr = 0
        elif y > limity2:
            dirr = 4
        elif y < limity1:
            dirr = 3
    elif y < limity2 and y > limity1:
        if x < limitx2 and x > limitx1:
            dirr = 0
        elif x > limitx2:
            dirr = 2
        elif x < limitx1:
            dirr = 1

    elif x < limitx1 and y < limity1:
        dirr = 5
    elif x > limitx2 and y < limity1:
        dirr = 6
    elif x < limitx1 and y > limity1:
        dirr = 7
    elif x > limitx2 and y > limity2:
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

def drone_stay_close(dir, mov, velocity1, velocity2):
    """Control velocities to track object"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    # Send velocities to move drone in different ways depending the number it gets
    if dir == 1:
        yaw_velocity = -velocity1
    elif dir == 2:
        yaw_velocity = velocity1
    elif dir == 3:
        up_down_velocity = velocity1
    elif dir == 4:
        up_down_velocity = -velocity1
    elif dir == 5:
        yaw_velocity = -velocity1
        up_down_velocity = velocity1
    elif dir == 6:
        yaw_velocity = velocity1
        up_down_velocity = velocity1
    elif dir == 7:
        yaw_velocity = -velocity1
        up_down_velocity = -velocity1
    elif dir == 8:
        yaw_velocity = velocity1
        up_down_velocity = -velocity1
    else:
        left_right_velocity = 0
        for_back_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0

    # If the drone is centered with the figure it can move back or forward depending the radius
    if dir == 0:
        if mov == 1:
            left_right_velocity = 0
            for_back_velocity = -velocity2
            up_down_velocity = 0
            yaw_velocity = 0
        elif mov == 2:
            left_right_velocity = 0
            for_back_velocity = velocity2
            up_down_velocity = 0
            yaw_velocity = 0
        else:
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0

    # Send the velocities to drone
    return left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity

def dinamic_speed(x, y):

    if (x < 500 and x > 230) and (y < 240 and y > 170):
        velocidad = 10
    elif (x < 730 and x > 500) and (y < 380 and y > 170):
        velocidad = 20
    elif (x < 730 and x > 460) and (y < 550 and y > 380):
        velocidad = 10
    elif (x < 460 and x > 230) and (y < 550 and y > 340):
        velocidad = 20

    elif (x < 884 and x > 230) and (y < 170 and y > 56):
        velocidad = 20
    elif (x < 884 and x > 730) and (y < 664 and y > 170):
        velocidad = 30
    elif (x < 730 and x > 230) and (y < 664 and y > 550):
        velocidad = 20
    elif (x < 230 and x > 76) and (y < 550 and y > 56):
        velocidad = 30

    elif x < 76 or x > 884 or y < 56 or y > 664:
        velocidad = 45

    else:
        velocidad = 30

    return velocidad





# Create an instance of Drone Tello
tello = Tello()
#Connect to Drone
tello.connect()
# Send message to drone to start stream
tello.streamon()

# Create some variables
counter = 0
dir = 0


# Wait 2 second to get respond of camera
time.sleep(2.0)
send_rc_control = False

while True:
    left_right_velocity = 0
    for_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    # Take a frame
    frame_read = tello.get_frame_read()
    # convert frame into an array
    frame = np.array(frame_read.frame)
    # process the image and return the center of the object detected and radius
    x, y, r, video = procesing(frame)
    # display grid in the video
    x_1, x_2, y_1, y_2, video_2 = display_grid(video)
    # display battery and logo in the video
    video_user = display_battery(display_text(video_2))
    # return a number that tell the drone how to move
    dir = track_drone_track(x, y, x_1, x_2, y_1, y_2)
    # return a number that tell the drone to move back or forward depending of the radius of the object
    mov = susana_distancia(r)
    speed = dinamic_speed(x, y)
    # Display information to the user
    print(f"x = {x} y = {y} r = {r}")
    print(f"dir = {dir} mov = {mov} counter = {counter}")

    # Drone Takeoff
    if counter == 40:
        tello.takeoff()
        send_rc_control = True

    left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity = drone_stay_close(dir, mov, speed, 20)
    # Send the velocities to drone
    time.sleep(1/30)
    if send_rc_control:
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

    counter = counter+1


    # Display the image
    cv2.imshow('Drone X', video_user)
    time.sleep(1 / 30)

    # Close video and finish the program
    if cv2.waitKey(5) & 0xFF == ord('q'):
        send_rc_control = False
        time.sleep(3.0)
        frame_read.stop()
        tello.stop_video_capture()
        cv2.destroyAllWindows()
        break

#Drone land
tello.land()