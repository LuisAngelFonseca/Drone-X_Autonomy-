from djitellopy import Tello
import cv2
import numpy as np
import time
import argparse
import imutils
import math
import os
import datetime
from numpy.core.multiarray import ndarray


# We declare the arguments when we run the script with the command line "-d" which execute the debug mode,
# he we can change the HSV values to calibrate the target HSV values
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-D', "--debug", action='store_true',
                    help='add the -D flag to enable debug HSV Mode, drone will act as a camera to improve HSV Calibration')
parser.add_argument('-ss', "--save_session", action='store_true',
                    help='add the -ss flag to save your sessions in order to have your tello sesions recorded')
args = parser.parse_args()

def callback(x):
    """ Function declare for the trackbars to work """
    pass

def stackImages(scale, imgArray):
    """ This function is for the debug mode that allows to have 4 images stack in the same window """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def display_grid(frame, size, x, y):
    """ Display grid on the frame and display two lines from the center to the tracking object """
    # You can give a value to size to change the grid sizes
    x1 = int(480 - (size))
    x2 = int(480 + (size))
    y1 = int(360 - (size * (3 / 4)) - 60)
    y2 = int(360 + (size * (3 / 4)) - 60)
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
    if x == None or y == None:
        x = 480
        y = 360
    # This part draw two lines from the center to the target
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(480, int(y)), color=(0, 255, 0), thickness=2)
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(int(x), 360), color=(0, 255, 0), thickness=2)

    return x1, x2, y1, y2, frame  # Return the position of each line and the frame

def display_text(frame_equ):
    """ Display text in the video """
    # Diplay text in the image
    font = cv2.FONT_ITALIC

    cv2.putText(frame_equ, text='Drone-X', org=(410, 25), fontFace=font, fontScale=1, color=(0, 0, 0),
                thickness=2, lineType=cv2.LINE_8)

    return frame_equ  # Return the frame with the text

def display_battery(frame_equ):
    """ Display a battery in the video that indicate the percentage of battery """
    # Display a battery in the image
    cv2.rectangle(frame_equ, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness=2)
    cv2.rectangle(frame_equ, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)

    global tiempo_elapsed, tiempo_actual, battery

    # Request battery every 15 seconds in autonomous mode
    if not args.debug:
        if tiempo_actual - tiempo_elapsed > 15:
            tiempo_elapsed = tiempo_actual
            print("Solicitar Bateria ")
            try:
                battery = int(tello.get_battery())  # Get battery level of the drone
            except:
                battery = 0
        else:
            # tiempo_elapsed = tiempo_elapsed
            tiempo_actual = int(time.time())
            #battery = battery
    # Request battery every 24 seconds in debug mode
    elif args.debug:
        if tiempo_actual - tiempo_elapsed > 24:
            tiempo_elapsed = tiempo_actual
            print("Solicitar Bateria Debug")
            try:
                battery = int(tello.get_battery())  # Get battery level of the drone
            except:
                battery = 0
        else:
            # tiempo_elapsed = tiempo_elapsed
            tiempo_actual = int(time.time())
            #battery = battery

    # Display a complete battery
    if battery > 75:
        cv2.rectangle(frame_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(frame_equ, pt1=(932, 9), pt2=(938, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(frame_equ, pt1=(940, 9), pt2=(947, 21), color=(0, 255, 0), thickness=-1)
    # Display a 2/3 of the battery
    elif battery < 75 and battery > 50:
        cv2.rectangle(frame_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 255), thickness=-1)
        cv2.rectangle(frame_equ, pt1=(932, 9), pt2=(940, 21), color=(0, 255, 255), thickness=-1)
    # Display 1/3 of the battery
    elif battery < 50 and battery > 25:
        cv2.rectangle(frame_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)

    return frame_equ

def object_detection(frame, lower_hsv, upper_hsv):
    """ Track the color in the frame """

    # If mode debug is active, make lower and upper parameters
    # be the ones from the trackbars
    if args.debug:
        color_lower = lower_hsv
        color_upper = upper_hsv

    # Else, use the lower and upper hardcoded parameters
    else:
        # Color range of wanted object
        color_lower = (90, 80, 55)
        color_upper = (107, 251, 255)
        # color_lower = (76, 96, 0)
        # color_upper = (129, 255, 255)

    # Blur frame, and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Construct a mask for the color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(frameHSV, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=5)

    # Find contours in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Initialize variables
    contours_circles = []  # Will contain contours with circularity
    center = None  # Will contain x and y coordinates of objects

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Go through every contour and check circularity,
        # if it falls between the range, append it
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))  # Formula for circularity
            if 0.85 < circularity < 1.05:
                contours_circles.append(contour)

    # Only proceed if at least one contour with circularity was found
    if len(contours_circles) > 0:
        # find the largest contour in the mask, then use
        # it to compute the radius and centroid
        max_contour = max(contours_circles, key=cv2.contourArea)
        radius = cv2.minEnclosingCircle(max_contour)[1]
        M = cv2.moments(max_contour)

        # Check if the computed radius meets a minimum size
        if radius > 5:
            # Object has been detected!, get center coordinates,
            # and draw a circle in the frame to visualize detection
            detection = True
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x, y = center
            # draw the circle and centroid on the frame
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #  No object has been detected
    else:
        detection = False
        x = None
        y = None
        radius = None

    return x, y, radius, detection, frame  # Return the position, radius of the object, and the frame

def drone_stay_close(x, y, limitx1, limitx2, limity1, limity2, r, distanceradius, tolerance):
    """
    Control velocities to track object, take x and y for the position of the target,
    limitx1 to limity2 are the lines of the center square of the grid
    r is the radius of the object
    distanceradius is the radius were you want the drone to stop
    tolerance is the range where the drone can keep the distance from the target
    """
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    # If the target is in the center, the drone can move back or forward
    if x < limitx2 and x > limitx1 and y < limity2 and y > limity1:
        for_back_velocity = int((distanceradius - r) * 1.3333)
        # If the drone is centered with the target and at the distance send velocities to 0
        if r < distanceradius + tolerance and r > distanceradius - tolerance:
            for_back_velocity = 0
    # Drone move to get the target centered
    else:
        yaw_velocity = int((x - 480) * .125)
        up_down_velocity = int((360 - y) * .1388888)
        for_back_velocity = 0

    # Send the velocities to drone
    return yaw_velocity, up_down_velocity, for_back_velocity


# Setup
# Create an instance of a Drone from the Tello library
tello = Tello()
# Variable to start the velocities control if True
send_rc_control = False
# Create a takeoff_timer for the takeoff and activate rc control
takeoff_timer = 0
# Frames per second of the stream
FPS = 25
# Frames per second for the video capture
FPS_vid = 10
# Set the speed for the override mode
S = 40
# Speed factor that increase the value for override
oSpeed = 1
# Battery value
battery = 0

# Values are given in pixels for following variables
# Grid size
grid_size = 100
# Radius of the object in which the drone will stop
radius_stop = 40
# Tolerance range in which the drone will stop
radius_stop_tolerance = 5

# Create 2 variables that count time
tiempo_actual = int(time.time())
tiempo_elapsed = int(time.time())

# Main Loop
while True:
    # Restore values to 0, to clean past values
    left_right_velocity = 0
    for_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    if not tello.connect():
        print("Tello not connected")

    if not tello.streamoff():
        print("Could not stop video stream")

    if not tello.streamon():
        print("Could not start video stream")

    # Capture a frame from drone camera
    frame_read = tello.get_frame_read()
    # Take the original frame for the video capture of the session
    frame_original = frame_read.frame

    # Infinite cycle to run code
    dron_continuos_cycle = True
    # You change the value with the spacebar and give you the manual control of the drone
    OVERRIDE = False
    # Check tello battery before starting
    battery = int(tello.get_battery())

    # This checks if we are in the debug mode,
    if args.debug:
        # Start debug mode, to calibrate de HSV values
        print("DEBUG MODE ENABLED!")

        # create trackbars for color change
        cv2.namedWindow("Color Calibration")
        cv2.resizeWindow('Color Calibration', 300, 350)
        # HUE
        cv2.createTrackbar('Hue Min', 'Color Calibration', 0, 179, callback)
        cv2.createTrackbar('Hue Max', 'Color Calibration', 179, 179, callback)
        # SATURATION
        cv2.createTrackbar('Sat Min', 'Color Calibration', 0, 255, callback)
        cv2.createTrackbar('Sat Max', 'Color Calibration', 255, 255, callback)
        # VALUES
        cv2.createTrackbar('Val Min', 'Color Calibration', 0, 255, callback)
        cv2.createTrackbar('Val Max', 'Color Calibration', 255, 255, callback)
        # ITERATIONS
        cv2.createTrackbar('Erosion', 'Color Calibration', 0, 30, callback)
        cv2.createTrackbar('Dilation', 'Color Calibration', 0, 30, callback)
    # Check if save session mode is on 
    if args.save_session:

        # If we are to save our sessions, we need to make sure the proper directories exist
        ddir = "Sessions"
        # If the provided path is not found, we create a new one
        if not os.path.isdir(ddir):
            os.mkdir(ddir)
        # We create a new folder path to save the actual video session with day and time
        ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':', '-').replace('.', '_'))
        os.mkdir(ddir)
        # Get the frame size 
        width = frame_original.shape[1]
        height = frame_original.shape[0]
        # We create two writer objects that create the video sessions 
        writer = cv2.VideoWriter("{}/TelloVideo.avi".format(ddir), cv2.VideoWriter_fourcc(*'XVID'),
                                 FPS_vid, (width, height))  
        writer_processed = cv2.VideoWriter("{}/TelloVideo_processed.avi".format(ddir), cv2.VideoWriter_fourcc(*'XVID'),
                                           FPS_vid, (width, height))  

    # This cycle is intended to run continuosly
    while dron_continuos_cycle:

        # Function that updates dron velocities in the override mode and autonomous mode 
        if send_rc_control:
            tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
        # If theres no frame, do not read the actual frame 
        if frame_read.stopped:
            frame_read.stop()
            break

        # Update frame 
        frame_original = frame_read.frame
        frame = frame_original.copy()
        
        # Update the track bars HSV mask and apply the erosion and dilation
        if args.debug:
            # Read all the track bars positions
            h_min = cv2.getTrackbarPos('Hue Min', 'Color Calibration')
            h_max = cv2.getTrackbarPos('Hue Max', 'Color Calibration')
            s_min = cv2.getTrackbarPos('Sat Min', 'Color Calibration')
            s_max = cv2.getTrackbarPos('Sat Max', 'Color Calibration')
            v_min = cv2.getTrackbarPos('Val Min', 'Color Calibration')
            v_max = cv2.getTrackbarPos('Val Max', 'Color Calibration')
            erosion = cv2.getTrackbarPos('Erosion', 'Color Calibration')
            dilation = cv2.getTrackbarPos('Dilation', 'Color Calibration')
            
            # Apply a Gaussian Blur to the image in order to reduce detail
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # Create HSV image, passing it from BGR
            frame_HSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            lower_hsv: ndarray = np.array([h_min, s_min, v_min])
            upper_hsv = np.array([h_max, s_max, v_max])

            mask = cv2.inRange(frame_HSV, lower_hsv, upper_hsv)
            mask = cv2.erode(mask, None, iterations=erosion)
            mask = cv2.dilate(mask, None, iterations=dilation)

            frameResult = cv2.bitwise_and(frame, frame, mask=mask)

            frameStack = stackImages(0.4, ([frame, frame_HSV], [mask, frameResult]))

            cv2.imshow('Stacked Images', frameStack)
        
        # Wait for a key to be press and grabs the value
        k = cv2.waitKey(20)

        # Drone Takeoff if the timer get to 40
        if takeoff_timer == 40:
            if not args.debug:
                print("Takeoff...")
                tello.get_battery()
                tello.takeoff()
            send_rc_control = True

        # Press T to take off in override mode 
        if k == ord('t') and takeoff_timer > 50:
            if not args.debug:
                print("Override mode: Takeoff...")
                tello.get_battery()
                tello.takeoff()

            send_rc_control = True

        # Press L to land
        if k == ord('l'):
            if not args.debug:
                print("Override mode: Land...")
                tello.land()
            send_rc_control = False

        # Press spacebar to enter override mode
        if k == 32:
            if not OVERRIDE:
                OVERRIDE = True
                print("OVERRIDE ENABLED")
            else:
                OVERRIDE = False
                print("OVERRIDE DISABLED")

        if OVERRIDE:
            # W to fly forward and S to fly back 
            if k == ord('w'):
                for_back_velocity = int(S * oSpeed)
            elif k == ord('s'):
                for_back_velocity = -int(S * oSpeed)
            else:
                for_back_velocity = 0

            # Z to fly clockwise and C to fly counter clockwise
            if k == ord('z'):
                yaw_velocity = int(S * oSpeed)
            elif k == ord('c'):
                yaw_velocity = -int(S * oSpeed)
            else:
                yaw_velocity = 0

            # Q to fly up and E to fly down
            if k == ord('q'):
                up_down_velocity = int(S * oSpeed)
            elif k == ord('e'):
                up_down_velocity = -int(S * oSpeed)
            else:
                up_down_velocity = 0

            # A to fly left and D to fly right
            if k == ord('a'):
                left_right_velocity = int(S * oSpeed)
            elif k == ord('d'):
                left_right_velocity = -int(S * oSpeed)
            else:
                left_right_velocity = 0

        # Press ESC to quit
        if k == 27:
            dron_continuos_cycle = False
            break

        # Take 4 points from the frame 
        pts1 = np.float32([[140, 0], [820, 0], [0, 666], [960, 660]])
        # Make the new screen size 
        pts2 = np.float32([[0, 0], [960, 0], [0, 720], [960, 720]])
        
        # Change the perspective of the frame to counteract the camera angle
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, matrix, (960, 720))

        # If we are in debug mode, send the track bars values to the object detection algorithm
        if args.debug:
            x, y, r, detection, frame_processed = object_detection(frame, lower_hsv, upper_hsv)
        else:
            x, y, r, detection, frame_processed = object_detection(frame, 0, 0)

        # Display grid in the actual frame
        x_1, x_2, y_1, y_2, frame_grid = display_grid(frame_processed, grid_size, x, y)

        # Display battery and logo in the video
        frame_user = display_battery(display_text(frame_grid))

        # Autonomous mode
        if send_rc_control and not OVERRIDE:

            if not args.debug:
                # Eliminate pass values
                left_right_velocity = 0
                for_back_velocity = 0
                up_down_velocity = 0
                yaw_velocity = 0

                if detection:
                    yaw_velocity, up_down_velocity, for_back_velocity = drone_stay_close(x, y, x_1, x_2, y_1,
                                                                                y_2, r, radius_stop, radius_stop_tolerance)
        # Save the video session if True
        if args.save_session:
            writer.write(frame_original)
            writer_processed.write(frame_user)

        # Display the video
        cv2.imshow('Drone X', frame_user)
        # Delay to showcase desired fps in video
        time.sleep(1 / FPS)
        # Update takeoff timer
        takeoff_timer += 1

    break

cv2.destroyAllWindows()

print("Goodbye")
tello.get_battery()
tello.end()
