from djitellopy import Tello
from collections import deque
import cv2
import numpy as np
import time
import argparse
import imutils
import math




def display_grid(frame, r):
    """Display grid on the video"""
    # Display each line of the dynamic grid
    x1 = int(480 - (r - 80))
    x2 = int(480 + (r - 80))
    y1 = int(300 - (r - 60))
    y2 = int(300 + (r - 60))
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
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

    return img_equ # Return the frame with the battery


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
    greenLower = (90, 80, 55)
    greenUpper = (107, 251, 255)

    # (29, 86, 6)
    # (64, 255, 255)
    # (90, 80, 55)
    # (107, 251, 255)

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
        radius = 300

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

def track_drone_track(x, y, limitx1, limitx2, limity1, limity2):
    """Return instruction on how to move to the drone based on position and the greed """
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

    return dirr # Return the instruction number from 0-8

def susana_distancia(r):
    """Return instruction from 0 - 2 to tello to stay close o fly away """
    if r > 220 and r < 240:
        mov = 0
    elif r > 240:
        mov = 1
    elif r < 220:
        mov = 2
    else:
        mov = 0

    return mov # Return an instruction from 0 - 2

def drone_stay_close(dir, mov, velocity1, velocity2):
    """Control velocities to track object"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    # Send velocities to move drone in different ways depending the number it gets
    if dir == 1:
        left_right_velocity = -velocity1
    elif dir == 2:
        left_right_velocity = velocity1
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

    # If the drone is centered with the object it can move back or forward depending the radius
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
    """
    Calculate the distance between the center and the position of the object
    then is multiplied by a factor that return a velocitie near 0 if the object is near the center of the screen
    and if the object is near the edges return a value near 70
    """
    velocidad = int((math.sqrt((x-480)**2+(y-300)**2))*(70/600))

    return velocidad


# Setup

tello = Tello()   # Create an instance of Drone Tello

tello.connect()   # Connect to Drone

tello.streamon()  # Send message to drone to start stream

counter = 0       # Create a counter for the takeoff and activate rc control
dir = 0           # This variable will send a number from 0 to 8 each number tell the drone how to move

#########################
#       #       #       #
#   5   #   3   #   6   #
#       #       #       #
#########################
#       #       #       #
#   1   #   0   #   2   #
#       #       #       #
#########################
#       #       #       #
#   7   #   4   #   8   #
#       #       #       #
#########################

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
    x_1, x_2, y_1, y_2, video_2 = display_grid(video, r) # Display grid in the actual frame, take video and radius of the object as arguments
                                                         # return the grid dynamic position first line passing through  x_1 ..... last line trough y_2

    video_user = display_battery(display_text(video_2))  # Display battery and logo in the video
    dir = track_drone_track(x, y, x_1, x_2, y_1, y_2)    # Takes the position of the object (x, y) and the center square of the grid
                                                         # to tell the drone how to move in 9 diferent cases(the big drawing tell what this function return)

    mov = susana_distancia(r)                            # Takes the radius of the object and return a number from 0 to 2
                                                         # 0 = don't move 1 = get away from object 2 = get closer from object

    speed = dinamic_speed(x, y)                          # Magic function that takes the position of the object and calculated the distance from the center
                                                         # return a velocity proportional to the distance 0-100

    print(f"x = {x} y = {y} r = {r} speed = {speed}")    # Display information to the user
    print(f"dir = {dir} mov = {mov} counter = {counter}")


    if counter == 40:
        #tello.takeoff()               # Drone Takeoff
        send_rc_control = True         # Turn on the rc control

    # Takes dir(0-8), mov(0-2), speed(0-70) and velocity_2 = 20 and return 4 velocities that will be send to the drone
    left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity = drone_stay_close(dir, mov, speed, 10)

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