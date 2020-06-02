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


def check_boundaries(value, tolerance, ranges, upper_or_lower):
    """ Returns value +- tolerance"""

    if ranges == 0:
        # Set the boundary for hue
        boundary = 179
    elif ranges == 1:
        # Set the boundary for saturation and value
        boundary = 255

    if upper_or_lower == 1:
        # In case the sum is greater than the parameter boundary, set it to boundary
        if value + tolerance > boundary:
            value = boundary
        else:
            value = value + tolerance
    else:
        # In case the subtraction is less than the parameter boundary, set it to boundary
        if value - tolerance < 0:
            value = 0
        else:
            value = value - tolerance

    return value


def pick_color(event, x, y, flags, params):
    """ Calculates HSV value from click and sets it to trackbars """

    frame, frame_HSV, stack, frameStack = params

    # Check if event was a mouse left click
    if event == cv2.EVENT_LBUTTONDOWN:

        # Transforms x and y coordinates of frameStack to original frame in upper left part of stack
        x = int(x*(frame.shape[1]/(frameStack.shape[1]/np.shape(stack)[1])))
        y = int(y*(frame.shape[0]/(frameStack.shape[0]/np.shape(stack)[0])))

        # Checks if x and y coordinates are inside the upper left frame
        if x <= frame.shape[1] and y <= frame.shape[0]:

            # Gets HSV values for pixel clicked
            pixel = frame_HSV[y, x]

            # HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
            # Set range = 0 for hue and range = 1 for saturation and brightness
            # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
            hue_lower = check_boundaries(pixel[0], 9, 0, 0)
            hue_upper = check_boundaries(pixel[0], 9, 0, 1)
            sat_lower = check_boundaries(pixel[1], 83, 1, 0)
            sat_upper = check_boundaries(pixel[1], 83, 1, 1)
            val_lower = check_boundaries(pixel[2], 100, 1, 0)
            val_upper = check_boundaries(pixel[2], 100, 1, 1)

            # Change trackbar position value to clicked one with tolerance
            cv2.setTrackbarPos('Hue Min', 'Color Calibration', hue_lower)
            cv2.setTrackbarPos('Hue Max', 'Color Calibration', hue_upper)
            cv2.setTrackbarPos('Sat Min', 'Color Calibration', sat_lower)
            cv2.setTrackbarPos('Sat Max', 'Color Calibration', sat_upper)
            cv2.setTrackbarPos('Val Min', 'Color Calibration', val_lower)
            cv2.setTrackbarPos('Val Max', 'Color Calibration', val_upper)


def text_instructions(frame):
    """ Draws instructions in the form of text on frame """

    # All position values are hardcoded, if size is changed, they will need to be readjusted
    cv2.putText(frame, 'Select color with mouse click', (30, 15), cv2.FONT_HERSHEY_COMPLEX,
                .5, (255, 255, 255))
    cv2.putText(frame, 'in upper left video to calibrate', (30, 35), cv2.FONT_HERSHEY_COMPLEX,
                .5, (255, 255, 255))
    cv2.putText(frame, '|', (330, 20), cv2.FONT_HERSHEY_COMPLEX,
                .7, (255, 255, 255))
    cv2.putText(frame, 'V', (326, 40), cv2.FONT_HERSHEY_COMPLEX,
                .7, (255, 255, 255))


def display_grid(frame, size, x, y):
    """ Display grid on the frame and display two lines from the center to the tracking object """
    # You can give a value to size to change the grid sizes
    x1 = int(480 - size)
    x2 = int(480 + size)
    y1 = int(360 - (size * (3 / 4)) - 60)
    y2 = int(360 + (size * (3 / 4)) - 60)
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
    if x is None or y is None:
        x = 480
        y = 300
    # This part draw two lines from the center to the target
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(480, int(y)), color=(0, 255, 0), thickness=2)
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(int(x), 300), color=(0, 255, 0), thickness=2)

    return x1, x2, y1, y2, frame  # Return the position of each line and the frame


def display_text(frame, text, org, color, blink=False):
    """ Display text in the video """
    global actual_time, elapsed_text_blink

    font = cv2.FONT_ITALIC
    #  Check if the text is needed to blink
    if blink:
        # This is to make the text blink one second on, one second off
        if 1 < actual_time - elapsed_text_blink < 2:
            cv2.putText(frame, text=text, org=org, fontFace=font, fontScale=1, color=color,
                        thickness=2, lineType=cv2.LINE_8)
        elif actual_time - elapsed_text_blink > 2:
            elapsed_text_blink = actual_time

    #  Display normal text is needed to blink
    else:
        cv2.putText(frame, text=text, org=org, fontFace=font, fontScale=1, color=color,
                    thickness=2, lineType=cv2.LINE_8)

    return frame  # Return the frame with the text


def display_icons(frame, bat=True, rec=False):
    """ Display icons in the video """
    global elapsed_time, actual_time, battery, elapsed_battery_blink, elapsed_recording_blink

    if bat:
        # Display a battery in the image representing its percentage
        cv2.rectangle(frame, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness=2)
        cv2.rectangle(frame, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)

        # Request battery every 15 seconds in autonomous mode
        if not args.debug:
            if actual_time - elapsed_time > 15:
                elapsed_time = actual_time
                print('Solicitar Bateria ')
                try:
                    battery = tello.get_battery()  # Get battery level of the drone
                    if not (battery == '' or battery == 'ok'):  # Checks if string battery is not empty
                        battery = int(battery)
                        print('Se convirtio valor de bateria a int')
                    else:
                        print('Bateria entrego "" o "ok"')
                except:
                    print('Error al pedir bateria')

        # Request battery every 24 seconds in debug mode
        elif args.debug:
            if actual_time - elapsed_time > 24:
                elapsed_time = actual_time
                print('Solicitar Bateria Debug')
                try:
                    battery = tello.get_battery()  # Get battery level of the drone
                    if not (battery == '' or battery == 'ok'):  # Checks if string battery is not empty
                        battery = int(battery)
                        print('Se convirtio valor de bateria a int')
                    else:
                        print('Bateria entrego "" o "ok"')
                except:
                    print('Error al pedir bateria')

        # Display a complete battery
        if battery > 75:
            cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 0), thickness=-1)
            cv2.rectangle(frame, pt1=(932, 9), pt2=(938, 21), color=(0, 255, 0), thickness=-1)
            cv2.rectangle(frame, pt1=(940, 9), pt2=(947, 21), color=(0, 255, 0), thickness=-1)
        # Display a 2/3 of the battery
        elif 75 > battery > 50:
            cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 255), thickness=-1)
            cv2.rectangle(frame, pt1=(932, 9), pt2=(940, 21), color=(0, 255, 255), thickness=-1)
        # Display 1/3 of the battery
        elif 50 > battery > 25:
            cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)
        # Display 1/3 of the battery blinking
        elif battery < 25:
            # Blinks battery every 0.5 seconds
            if 0.5 < actual_time - elapsed_battery_blink < 1:
                cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)
            elif actual_time - elapsed_battery_blink > 1:
                elapsed_battery_blink = actual_time

    if rec:
        # Blinks battery every 1 seconds
        if 1 < actual_time - elapsed_recording_blink < 2:
            cv2.circle(frame, (890, 15), 10, (0, 0, 255), -1)  # Put a red circle indicating its recording
        elif actual_time - elapsed_recording_blink > 2:
            elapsed_recording_blink = actual_time

    return frame


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
    # a series of erodes and dilates to remove any small
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


def drone_stay_close(x, y, limit_x1, limit_x2, limit_y1, limit_y2, radius, distance_radius, tolerance):
    """
    Control velocities to track object, take x and y for the position of the target,
    limit_x1 to limit_y2 are the lines of the center square of the grid
    r is the radius of the object
    distance_radius is the radius were you want the drone to stop
    tolerance is the range where the drone can keep the distance from the target
    """
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    # If the target is in the center, the drone can move back or forward
    if limit_x2 > x > limit_x1 and limit_y2 > y > limit_y1:
        for_back_velocity = int((distance_radius - radius) * 1.3333)
        # If the drone is centered with the target and at the distance send velocities to 0
        if distance_radius + tolerance > radius > distance_radius - tolerance:
            for_back_velocity = 0
    # Drone move to get the target centered
    else:
        yaw_velocity = int((x - 480) * .125)
        up_down_velocity = int((300 - y) * .2)
        for_back_velocity = 0

    # Send the velocities to drone
    return yaw_velocity, up_down_velocity, for_back_velocity


def get_frames_while_flying(rmax, rmin, stops):
    list_of_stops = []
    for stop in range(stops + 1):
        stop = (((rmax - rmin) / stops) * (stop)) + rmin
        list_of_stops.append(stop)

    return list_of_stops


def count_tomatoes(frame_read, frame_user):
    # Declaration of color variables
    # Red variables
    color_lower = (15, 100, 55)
    color_upper = (34, 255, 255)

    # Blur frame, and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame_read, (11, 11), 0)
    frameHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Construct a mask for the color, then perform
    # a series of erodes and dilates to remove any small
    # blobs left in the mask
    mask = cv2.inRange(frameHSV, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    t_cnts = []

    # Only will show contours if at least one blue contour is found
    if len(contours) > 0:
        # loop over the contours
        for contour in contours:
            # compute the center of the contour
            area = cv2.contourArea(contour)
            if area > 300:
                t_cnts.append(contour)
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # # draw the contour and center of the shape on the image
                # cv2.drawContours(frame_user, [c], -1, (0, 255, 0), 2)
                # cv2.circle(frame_user, (cX, cY), 7, (255, 255, 255), -1)

    return len(t_cnts), frame_user


# Setup
# Create an instance of a Drone from the Tello library
tello = Tello()
# Create a taking off boolean
is_taking_off = False
# Create a landing boolean
is_landing = False
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
grid_size = 60
# Radius of the object in which the drone will stop
radius_stop = 60
# Tolerance range in which the drone will stop
radius_stop_tolerance = 5

# Maximum radius of target that the drone will detect
max_radius = 45
# Minimum radius of target that the drone will detect
min_radius = 10
# Number of frames to save in range of the radius
frames_to_capture = 3
# Create a list to save a frame at different distances
frame_capture_list = get_frames_while_flying(max_radius, min_radius, frames_to_capture)

# Create variables that counts time
actual_time = time.time()
elapsed_time = actual_time
# Create variables that count time to blink text
elapsed_text_blink = actual_time
# Create variables that count time to blink battery when low
elapsed_battery_blink = actual_time
# Create variables that count time to blink recording icon
elapsed_recording_blink = actual_time

# Number of tomatoes found
tomatoes = 0

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<  DRONE CYCLE ->->->->->->->->->->->->->->->->->->->->->
while True:

    # --------------------------- TELLO SETUP SECTION -----------------------------
    # Restore values to 0, to clean past values
    left_right_velocity = 0
    for_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    if not tello.connect():
        print('Tello not connected')

    if not tello.streamoff():
        print('Could not stop video stream')

    if not tello.streamon():
        print('Could not start video stream')

    # --------------------------- VARIABLE DECLARATION SECTION -----------------------------
    # Infinite cycle to run code
    drone_continuous_cycle = True
    # You change the value with the spacebar and give you the manual control of the drone
    OVERRIDE = False
    # Check tello battery before starting
    print('Solicitar Bateria ')
    try:
        battery = tello.get_battery()  # Get battery level of the drone
        if not (battery == '' or battery == 'ok'):  # Checks if string battery is not empty
            battery = int(battery)
            print('Se convirtio valor de bateria a int')
        else:
            print('Bateria entrego "" o "ok"')
    except:
        print('Error al pedir bateria')

    # --------------------------- DEBUG TRACKBAR SECTION -----------------------------
    # This checks if we are in the debug mode,
    if args.debug:
        # Start debug mode, to calibrate de HSV values
        print('DEBUG MODE ENABLED!')

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

    # --------------------------- SAVE SESSION SECTION -----------------------------
    # Capture a frame from drone camera
    frame_read = tello.get_frame_read()
    # Take the original frame for the video capture of the session
    frame_original = frame_read.frame
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

    # -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-< CONTINUOUS DRONE CYCLE ->->->->->->->->->->->->->->->->->->->->->
    while drone_continuous_cycle:

        # --------------------------- SEND DRONE VELOCITY SECTION -----------------------------
        # Function that updates drone velocities in the override mode and autonomous mode
        if tello.is_flying:
            tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

        # --------------------------- FRAME READ SECTION -----------------------------
        # If there is no frame, do not read the actual frame
        if frame_read.stopped:
            frame_read.stop()
            break

        # Update frame
        frame_original = frame_read.frame
        frame = frame_original.copy()

        # --------------------------- DEBUG CALIBRATION SECTION -----------------------------
        # Update the trackbars HSV mask and apply the erosion and dilation
        if args.debug:
            # Read all the trackbars positions
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

            stack = ([frame, frame_HSV], [mask, frameResult])
            frameStack = stackImages(0.4, stack)

            text_instructions(frameStack)
            cv2.imshow('Stacked Images', frameStack)

            cv2.setMouseCallback('Stacked Images', pick_color, (frame, frame_HSV, stack, frameStack))

        # --------------------------- FRAME PROCESSING SECTION -----------------------------
        # Take 4 points from the frame
        pts1 = np.float32([[140, 0], [820, 0], [0, 666], [960, 660]])
        # Make the new screen size
        pts2 = np.float32([[0, 0], [960, 0], [0, 720], [960, 720]])

        # Change the perspective of the frame to counteract the camera angle
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frame_perspective = cv2.warpPerspective(frame, matrix, (960, 720))

        # If we are in debug mode, send the trackbars values to the object detection algorithm
        if args.debug:
            x, y, r, detection, frame_processed = object_detection(frame_perspective, lower_hsv, upper_hsv)
        else:
            x, y, r, detection, frame_processed = object_detection(frame_perspective, 0, 0)

        # Display grid in the actual frame
        x_1, x_2, y_1, y_2, frame_grid = display_grid(frame_processed, grid_size, x, y)

        # Display battery, logo and mode in the video
        frame_user = display_icons(display_text(frame_grid, 'Drone-x', (410, 25), (0, 0, 0)), bat=True)
        if OVERRIDE:
            frame_user = display_text(frame_user, 'OVERRIDE MODE: ON', (5, 710), (0, 0, 255), blink=True)
        if args.debug:
            frame_user = display_text(frame_user, 'DEBUG MODE: ON', (5, 25), (0, 0, 255), blink=True)

        # --------------------------- READ KEY SECTION -----------------------------
        # Wait for a key to be press and grabs the value
        k = cv2.waitKey(20)

        # Press ESC to quit -!-!-!-> EXIT PROGRAM <-!-!-!-
        if k == 27:
            if tello.is_flying:
                for i in range(50):
                    tello.send_rc_control(0, 0, 0, 0)  # Stop the drone if it has momentum
                    time.sleep(1 / FPS)
            drone_continuous_cycle = False
            break

        if is_taking_off:
            tello.get_battery()
            tello.takeoff()
            is_taking_off = False

        # Press T to take off
        if (k == ord('t') or k == ord('T')) and not tello.is_flying and not args.debug:
            is_taking_off = True
            print('Takeoff...')
            frame_user = display_text(frame_user, 'Taking off...', (5, 25), (0, 255, 255))

        if is_landing:
            for i in range(50):
                tello.send_rc_control(0, 0, 0, 0)  # Stop the drone if it has momentum
                time.sleep(1 / FPS)
            tello.land()
            is_landing = False

        # Press L to land
        if (k == ord('l') or k == ord('L')) and tello.is_flying and not args.debug:
            is_landing = True
            print('Land...')
            frame_user = display_text(frame_user, 'Landing...', (5, 25), (0, 255, 255))

        # Press spacebar to enter override mode
        if k == 32 and tello.is_flying:
            if not OVERRIDE:
                OVERRIDE = True
                print('OVERRIDE ENABLED')
            else:
                OVERRIDE = False
                print('OVERRIDE DISABLED')

        if OVERRIDE:
            # W to fly forward and S to fly back
            if k == ord('w') or k == ord('W'):
                for_back_velocity = int(S * oSpeed)
            elif k == ord('s') or k == ord('S'):
                for_back_velocity = -int(S * oSpeed)
            else:
                for_back_velocity = 0

            #  C to fly clockwise and Z to fly counter clockwise
            if k == ord('c') or k == ord('C'):
                yaw_velocity = int(S * oSpeed)
            elif k == ord('z') or k == ord('Z'):
                yaw_velocity = -int(S * oSpeed)
            else:
                yaw_velocity = 0

            # Q to fly up and E to fly down
            if k == ord('q') or k == ord('Q'):
                up_down_velocity = int(S * oSpeed)
            elif k == ord('e') or k == ord('E'):
                up_down_velocity = -int(S * oSpeed)
            else:
                up_down_velocity = 0

            # A to fly left and D to fly right
            if k == ord('d') or k == ord('D'):
                left_right_velocity = int(S * oSpeed)
            elif k == ord('a') or k == ord('A'):
                left_right_velocity = -int(S * oSpeed)
            else:
                left_right_velocity = 0

        # --------------------------- AUTONOMOUS SECTION -----------------------------
        if tello.is_flying and not OVERRIDE and not args.debug:
            # Eliminate pass values
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0

            if detection:
                yaw_velocity, up_down_velocity, for_back_velocity = drone_stay_close(x, y, x_1, x_2, y_1,
                                                                                     y_2, r, radius_stop,
                                                                                     radius_stop_tolerance)

        # --------------------------- WRITE VIDEO SESSION SECTION -----------------------------
        # Save the video session if True
        if args.save_session:
            frame_user = display_text(frame_user, 'REC', (810, 25), (0, 0, 0))
            frame_user = display_icons(frame_user, bat=True, rec=True)
            writer.write(frame_original)
            writer_processed.write(frame_user)

        # --------------------------- TOMATO COUNTER SECTION -----------------------------
        # Get the number of tomatoes 
        # if r is not None and frame_capture_list:
            # if (frame_capture_list[0] + 5) > r > (frame_capture_list[0] - 5):
            #     del frame_capture_list[0]
            # if 35 < r < 45:
        tomatoes = count_tomatoes(frame_original, 0)[0]
        frame_user = count_tomatoes(frame_original, frame_user)[1]
        print(tomatoes)
        # --------------------------- SHOW VIDEO SECTION -----------------------------
        # Display the video

        cv2.imshow('Drone X', frame_user)

        # --------------------------- MISCELLANEOUS SECTION -----------------------------
        # Save actual time
        actual_time = time.time()
        # Delay to showcase desired fps in video
        time.sleep(1 / FPS)

    break

cv2.destroyAllWindows()
print('Goodbye')
tello.get_battery()
tello.end()
