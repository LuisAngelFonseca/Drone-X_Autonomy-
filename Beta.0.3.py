from djitellopy import Tello
from collections import deque
import cv2
import numpy as np
import time
import argparse
import imutils
import math
import os
import datetime
from numpy.core.multiarray import ndarray

"""
Esta parte esta encargada de los argumetnos que se brindan cuando corres este script desde la linea de comando
al agregarle el -D se corre en version de debugeo donde se pueden cambiar los valores de hsv de acorde a la camara
del dron
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-D', "--debug", action='store_true',
                    help='add the -D flag to enable debug HSV Mode, drone will act as a camera to improve HSV Calibration')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your sessions in order to have your tello sesions recorded')

args = parser.parse_args()

# Speed of the drone
S = 40
# Factor de velocidad
oSpeed = 1

def callback(x):
    pass

def stackImages(scale, imgArray):
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
    """Display grid on the video"""
    # Display each line of the dynamic grid
    x1 = int(480 - (size))
    x2 = int(480 + (size))
    y1 = int(360 - (size*(3/4))-60)
    y2 = int(360 + (size*(3/4))-60)
    cv2.line(frame, pt1=(x1, 0), pt2=(x1, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(x2, 0), pt2=(x2, 720), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y1), pt2=(960, y1), color=(255, 0, 0), thickness=2)
    cv2.line(frame, pt1=(0, y2), pt2=(960, y2), color=(255, 0, 0), thickness=2)
    if x == None or y == None:
        x = 480
        y = 360
    cv2.line(frame, pt1=(int(x),int(y)), pt2=(480, int(y)), color=(0, 255, 0), thickness=2)
    cv2.line(frame, pt1=(int(x), int(y)), pt2=(int(x), 360), color=(0, 255, 0), thickness=2)

    return x1, x2, y1, y2, frame # Return the position of each line and the frame

def display_text(img_equ):
    """Display text in the video"""
    # Diplay text in the image
    font = cv2.FONT_ITALIC

    cv2.putText(img_equ, text='Drone-X', org=(410, 25), fontFace=font, fontScale=1, color=(0, 0, 0),
                thickness=2, lineType=cv2.LINE_8)

    return img_equ  # Return the frame with the text

def display_battery(img_equ):
    """Display a battery in the video that indicate the percentage of battery"""
    # Display a battery in the image
    cv2.rectangle(img_equ, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness=2)
    cv2.rectangle(img_equ, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)

    global tiempo_elapsed, tiempo_actual, battery

    # la primera vez la bateria es 100, pero este valor solo dura 5 segundos
    battery = 100
    if not args.debug:
        if tiempo_actual - tiempo_elapsed > 6:
            tiempo_elapsed = tiempo_actual
            print("Solicitar Bateria ")
            try:
                battery = int(tello.get_battery())  # Get battery level of the drone
            except:
                battery = 0
        else:
            tiempo_elapsed = tiempo_elapsed
            tiempo_actual = int(time.time())
            battery = battery
    elif args.debug:
        if tiempo_actual - tiempo_elapsed > 24:
            tiempo_elapsed = tiempo_actual
            print("Solicitar Bateria Debug")
            try:
                battery = int(tello.get_battery())  # Get battery level of the drone
            except:
                battery = 0
        else:
            tiempo_elapsed = tiempo_elapsed
            tiempo_actual = int(time.time())
            battery = battery

    # Display a complete battery
    if battery > 75:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(938, 21), color=(0, 255, 0), thickness=-1)
        cv2.rectangle(img_equ, pt1=(940, 9), pt2=(947, 21), color=(0, 255, 0), thickness=-1)
    # Display a 2/3 of the battery
    elif battery < 75 and battery > 50:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 255), thickness=-1)
        cv2.rectangle(img_equ, pt1=(932, 9), pt2=(940, 21), color=(0, 255, 255), thickness=-1)
    # Display 1/3 of the battery
    elif battery < 50 and battery > 25:
        cv2.rectangle(img_equ, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)

    return img_equ

def processing(frame, lower_hsv, upper_hsv):
    """Track the color in the frame"""

    if args.debug:
        color_lower = lower_hsv
        color_upper = upper_hsv
    else:
        # Color range of wanted object
        color_lower = (90, 80, 55)
        color_upper = (107, 251, 255)

    # blur it, and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frameHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(frameHSV, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    contours_circles = []

    # only proceed if at least one contour was found
    if len(contours) > 0:
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if 0.85 < circularity < 1.05:
                contours_circles.append(contour)

    if len(contours_circles) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        max_contour = max(contours_circles, key=cv2.contourArea)
        radius = cv2.minEnclosingCircle(max_contour)[1]
        M = cv2.moments(max_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x, y = center

        # only proceed if the radius meets a minimum size
        if radius > 15:
            # draw the circle and centroid on the frame
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        detection = True
    else:
        detection = False
        x = None
        y = None
        radius = None

    return x, y, radius, detection, frame  # Return the position and radius of the object and also the frame

def drone_stay_close(x, y, limitx1, limitx2, limity1, limity2, r,  distanceradius, tolerance):
    """Control velocities to track object"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity, bus_stop

    if x < limitx2 and x > limitx1 and y < limity2 and y > limity1:
        for_back_velocity = int((distanceradius - r) *.25)
        bus_stop = False
        if r < distanceradius + tolerance and r > distanceradius - tolerance:
            for_back_velocity = 0
            bus_stop = True
    else:
        yaw_velocity = int((x-480)*.125)
        up_down_velocity = int((360-y)*.1388888)
        for_back_velocity = 0
        bus_stop = False

    # Send the velocities to drone
    return yaw_velocity, up_down_velocity, for_back_velocity, bus_stop

def drone_microbusero(rmin, rmax, stops):
    list_stops = []
    for stop in range(stops+1):
        list_stops.append((((rmax-rmin)/stops)*stop)+rmin)
    return (list_stops)


# Setup
# Create an instance of Drone Tello
tello = Tello()

# Connect to Drone
tello.connect()

# Send message to drone to start stream
tello.streamon()

send_rc_control = False

# Create a counter for the takeoff and activate rc control
counter = 0
counter_stop = 0

# Frames per second
FPS = 25

# Create 2 variables that count time
tiempo_actual = int(time.time())
tiempo_elapsed = int(time.time())

#
init_time = int(time.time())
turn_time = 0

stop_value = drone_microbusero(185,250,3)
stop = stop_value[1]
i = 1
while True:
    # Restore values to 0, to clean past values
    left_right_velocity = 0
    for_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0
    speed = 10

    if not tello.connect():
        print("Tello not connected")

    if not tello.set_speed(speed):
        print("Not set speed to lowest possible")

    if not tello.streamoff():
        print("Could not stop video stream")

    if not tello.streamon():
        print("Could not start video stream")

    # Capture a frame from drone camera
    frame_read = tello.get_frame_read()

    # esta variable se encarga de decidir cuando corre el main verdadero y cuando no
    #Lo que esta afuera del while true solo correra una vez
    Ciclo_Dron = False
    frame_Count = 0
    # esta variable hace que puedas controlar al dron con la barra espaciadora
    OVERRIDE = False
    #check tello battery
    tello.get_battery()

    # This checks if we are in the debug mode,
    if args.debug:
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

    if args.save_session:

    # If we are to save our sessions, we need to make sure the proper directories exist
        ddir = "Sessions"

        if not os.path.isdir(ddir):
                os.mkdir(ddir)
        ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':', '-').replace('.', '_'))
        os.mkdir(ddir)

        cap = frame_read.frame

        width = cap.shape[1]
        height = cap.shape[0]

        print(width,height)
        print(cap.shape)

        writer = cv2.VideoWriter("{}/TelloVideo_processed.avi".format(ddir), cv2.VideoWriter_fourcc(*'XVID'),
                                 30, (width, height))  # con esta wea se guardan videos
        writer_proccesed = cv2.VideoWriter("{}/TelloVideo.avi".format(ddir), cv2.VideoWriter_fourcc(*'XVID'),
                                 30, (width, height))  # con esta wea se guardan videos

    # aqui van las cosas que irian en el main normal
    while not Ciclo_Dron:

        # Function that updates dron speeds
        if send_rc_control:
            tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

        if frame_read.stopped:
            frame_read.stop()
            break

        # Frame read
        frame = frame_read.frame

        frame_Count += 1

        if args.debug:
            #Read all the trackbars positions
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
            #Create HSV image, passing it from BGR
            frame_HSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            lower_hsv: ndarray = np.array([h_min, s_min, v_min])
            upper_hsv = np.array([h_max, s_max, v_max])

            mask = cv2.inRange(frame_HSV, lower_hsv, upper_hsv)
            mask = cv2.erode(mask, None, iterations=erosion)
            mask = cv2.dilate(mask, None, iterations=dilation)

            frameResult = cv2.bitwise_and(frame, frame, mask=mask)

            frameStack = stackImages(0.4, ([frame, frame_HSV], [mask, frameResult]))

            cv2.imshow('Stacked Images', frameStack)

        # Delay to showcase desired fps in video
        time.sleep(1 / FPS)

        k = cv2.waitKey(20)

        # Drone Takeoff
        if counter == 40:
            if not args.debug:
                print("Preparado pa la guerra")
                tello.takeoff()
                tello.get_battery()
            send_rc_control = True

        # Press T to take off
        if k == ord('t') and counter > 50:
            if not args.debug:
                print("Preparado la guerra manual")
                tello.takeoff()
                tello.get_battery()
            send_rc_control = True

        # Press L to land
        if k == ord('l'):
            if not args.debug:
                print("A mimir")
                tello.land()
            send_rc_control = False

        # Press Backspace for controls override
        if k == 32:
            if not OVERRIDE:
                OVERRIDE = True
                print("OVERRIDE ENABLED")
            else:
                OVERRIDE = False
                print("OVERRIDE DISABLED")

        if OVERRIDE:
            # S & W to fly forward & back
            if k == ord('w'):
                for_back_velocity = int(S * oSpeed)
            elif k == ord('s'):
                for_back_velocity = -int(S * oSpeed)
            else:
                for_back_velocity = 0

            # a & d to pan left & right
            if k == ord('z'):
                yaw_velocity = int(S * oSpeed)
            elif k == ord('c'):
                yaw_velocity = -int(S * oSpeed)
            else:
                yaw_velocity = 0

            # Q & E to fly up & down
            if k == ord('e'):
                up_down_velocity = int(S * oSpeed)
            elif k == ord('q'):
                up_down_velocity = -int(S * oSpeed)
            else:
                up_down_velocity = 0

            # c & z to fly left & right
            if k == ord('a'):
                left_right_velocity = int(S * oSpeed)
            elif k == ord('d'):
                left_right_velocity = -int(S * oSpeed)
            else:
                left_right_velocity = 0

        # Salir
        if k == 27:
            Main_Real = True
            break

        pts1 = np.float32([[140, 0],[820, 0],[0, 666],[960,660]])
        pts2 = np.float32([[0, 0], [960, 0], [0,720], [960, 720]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, matrix, (960,720))

        # Getting the position of the object, radius and tracking the object in the frame
        if args.debug:
            x, y, r, detection, video = processing(frame,lower_hsv,upper_hsv)
        else:
            x, y, r, detection, video = processing(frame,0,0)

        # Display grid in the actual frame, take video and radius of the object as arguments
        # return the grid dynamic position first line passing through  x_1 ..... last line trough y_2
        x_1, x_2, y_1, y_2, video_2 = display_grid(video, 100, x, y)

        # display battery and logo in the video
        video_user = display_battery(display_text(video_2))

        if send_rc_control and not OVERRIDE:

            if not args.debug:

                left_right_velocity = 0
                for_back_velocity = 0
                up_down_velocity = 0
                yaw_velocity = 0

                if detection:
                    yaw_velocity, up_down_velocity, for_back_velocity, bus_stop = drone_stay_close(x, y, x_1, x_2, y_1, y_2, r, stop, 5)
                    print(f"x = {x} y = {y} r = {r}")
                    if bus_stop:
                        counter_stop = counter_stop + 1
                        print(f"counter = {counter} counter stop = {counter_stop} stop = {stop}")
                if counter_stop > 20:
                    i += 1
                    if i < len(stop_value) - 1:
                        stop = stop_value[i]
                    counter_stop = 0
                    bus_stop = False
                    print(f"counter = {counter} counter stop = {counter_stop} r stop = {stop} bus = {bus_stop}")

                # Display information to the user
                print(f"counter = {counter}")
        # Update counter
        counter = counter + 1
        # Display the video
        cv2.imshow('Drone X', video_user)

        if args.save_session:

            writer.write(frame)
            writer_proccesed.write(video_user)

    break

cv2.destroyAllWindows()

print("Adios Vaquero")
tello.get_battery()

tello.end()
