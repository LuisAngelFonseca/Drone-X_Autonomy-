from djitellopy import Tello
from collections import deque
import cv2
import numpy as np
import time
import argparse
import imutils
import math

def display_grid(frame, size, x, y):
    """Display grid on the video"""
    # Display each line of the dynamic grid
    x1 = int(480 - (size))
    x2 = int(480 + (size))
    y1 = int(360 - (size*(3/4)))
    y2 = int(360 + (size*(3/4)))
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(int(x),int(y)), pt2=(480, int(y)), color=(0, 255, 0), thickness=2)
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(int(x), 360), color=(0, 255, 0), thickness=2)

    return x1, x2, y1, y2, frame # Return the position of each line and the frame

def display_text(img_equ):
    """Display text in the video"""
    # Diplay text in the image
    font = cv2.FONT_ITALIC

    cv2.putText(img_equ, text='Drone-X', org=(410, 25), fontFace=font, fontScale=1, color=(0, 0, 0),
                thickness=2, lineType=cv2.LINE_8)

    return img_equ #Return the frame with the text

def display_battery(img_equ):
    """Display a battery in the video that indicate the percentage of battery"""
    # Display a battery in the image
    cv2.rectangle(img_equ, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness= 2)
    cv2.rectangle(img_equ, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)
    try:
        battery = int(tello.get_battery()) #Get battery level of the drone
    except:
        battery = 0


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

    return img_equ # Return the frame with the batteryqqqqqq

def procesing(frame):
    """Track the color in the frame"""
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

    # (29, 86, 6)
    # (64, 255, 255)
    # (90, 80, 55)
    # (107, 251, 255)

    pts = deque(maxlen=args["buffer"])

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)

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
        radius = 40


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

    return(x, y, radius, frame) # Return the position and radius of the object and also the frame

def drone_stay_close(x, y, limitx1, limitx2, limity1, limity2, r,  distanceradius, tolerance):
    """Control velocities to track object"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    if x < limitx2 and x > limitx1 and y < limity2 and y > limity1:
        for_back_velocity = int((distanceradius - r) * 1.33333)
        if r < distanceradius + tolerance and r > distanceradius - tolerance:
            for_back_velocity = 0
    else:
        yaw_velocity = int((x-480)*.125)
        up_down_velocity = int((360-y)*.1388888)

    # Send the velocities to drone
    return yaw_velocity, up_down_velocity, for_back_velocity


# Setup

tello = Tello()   # Create an instance of Drone Tello

tello.connect()   # Connect to Drone

tello.streamon()  # Send message to drone to start stream

counter = 0       # Create a counter for the takeoff and activate rc control

time.sleep(2.0)   # Wait 2 second to get respond of camera
send_rc_control = False # This variable is false until we want to send rc control commands to drone, this is after the takeoff

# Main
while True:
    # Restore values to 0, to clean past values
    left_right_velocity = 0
    for_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    frame_read = tello.get_frame_read()                  # Capture a frame from drone camera
    frame = np.array(frame_read.frame)                   # Frame turn into an array
    x, y, r, video = procesing(frame)                    # Getting the position of the object, radius and tracking the object in the frame
    x_1, x_2, y_1, y_2, video_2 = display_grid(video, 100, x, y)  # Display grid in the actual frame, take video and radius of the object as arguments
                                                                  # return the grid dynamic position first line passing through  x_1 ..... last line trough y_2

    video_user = display_battery(display_text(video_2))  # Display battery and logo in the video

    print(f"x = {x} y = {y} r = {r} ")    # Display information to the user
    print(f"counter = {counter}")


    if counter == 40:
        tello.takeoff()               # Drone Takeoff
        send_rc_control = True         # Turn on the rc control

    yaw_velocity, up_down_velocity, for_back_velocity = drone_stay_close(x, y, x_1, x_2, y_1, y_2, r, 40, 5)

    time.sleep(1/30)                  # Delay
    if send_rc_control:               # If true, we send 4 velocities to drone(each velocity can take de value from -100 to 100)
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

    counter = counter+1               # Update counter

    cv2.imshow('Drone X', video_user) # Display the video
    time.sleep(1 / 30)                # Delay

    # Close video and finish the program
    if cv2.waitKey(5) & 0xFF == ord('q'):
        send_rc_control = False
        frame_read.stop()
        tello.stop_video_capture()
        cv2.destroyAllWindows()
        time.sleep(3.0)
        break

tello.land()                        #Drone land