from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils
import math
import os
import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import sys
import pickle


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
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                         scale)
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
        x = int(x * (frame.shape[1] / (frameStack.shape[1] / np.shape(stack)[1])))
        y = int(y * (frame.shape[0] / (frameStack.shape[0] / np.shape(stack)[0])))

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


class DroneX:

    def __init__(self):

        # ----- CONSTANTS -----
        # Speed of the drone
        self.S = 40
        # Speed factor that increase the value for override
        self.oSpeed = 1
        # Battery value
        self.battery = 0

        # Frames per second of the pygame window display
        self.FPS = 30
        # Frames per second of the stream
        self.FPS = 25
        # Frames per second for the video capture
        self.FPS_vid = 10

        # Values are given in pixels for following variables
        # Grid size
        self.grid_size = 60
        # Radius of the object in which the drone will stop
        self.radius_stop = 60
        # Tolerance range in which the drone will stop
        self.radius_stop_tolerance = 5

        # Create variables that counts time
        self.actual_time = time.time()
        self.elapsed_time = self.actual_time
        # Create variables that count time to blink text
        self.elapsed_text_blink = self.actual_time
        # Create variables that count time to blink battery when low
        self.elapsed_battery_blink = self.actual_time
        # Create variables that count time to blink recording icon
        self.elapsed_recording_blink = self.actual_time

        # ----- VARIABLES -----
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.__event = 0

        self.is_landing = False
        self.is_taking_off = False
        self.drone_continuous_cycle = False
        self.OVERRIDE = False
        self.debug = False
        self.save_session = False

        self.frame_read = None
        self.writer = None
        self.writer_processed = None

    def get_event(self):
        return self.__event

    def set_event(self, val):
        self.__event = val

    def initializer(self):
        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        self.drone_continuous_cycle = True

        # --------------------------- VARIABLE DECLARATION SECTION -----------------------------

        # Check tello battery before starting
        print('Solicitar Bateria ')
        try:
            self.battery = self.tello.get_battery()  # Get battery level of the drone
            if not (self.battery == '' or self.battery == 'ok'):  # Checks if string battery is not empty
                self.battery = int(self.battery)
                print('Se convirtio valor de bateria a int')
            else:
                print('Bateria entrego "" o "ok"')
        except:
            print('Error al pedir bateria')

        # --------------------------- DEBUG TRACKBAR SECTION -----------------------------
        # This checks if we are in the debug mode,
        if self.debug:
            # Start debug mode, to calibrate de HSV values
            print('DEBUG MODE ENABLED!')
            # create trackbars for color change
            cv2.namedWindow("Color Calibration")
            cv2.resizeWindow('Color Calibration', 400, 400)
            # HUE
            cv2.createTrackbar('Hue Min', 'Color Calibration', 0, 179, self.callback)
            cv2.createTrackbar('Hue Max', 'Color Calibration', 179, 179, self.callback)
            # SATURATION
            cv2.createTrackbar('Sat Min', 'Color Calibration', 0, 255, self.callback)
            cv2.createTrackbar('Sat Max', 'Color Calibration', 255, 255, self.callback)
            # VALUES
            cv2.createTrackbar('Val Min', 'Color Calibration', 0, 255, self.callback)
            cv2.createTrackbar('Val Max', 'Color Calibration', 255, 255, self.callback)
            # ITERATIONS
            cv2.createTrackbar('Erosion', 'Color Calibration', 1, 30, self.callback)
            cv2.createTrackbar('Dilation', 'Color Calibration', 1, 30, self.callback)

            with open('mask_values.pkl', 'rb') as f:
                color_lower, color_upper = pickle.load(f)
                f.close()

            hue_lower, sat_lower, val_lower = color_lower
            hue_upper, sat_upper, val_upper = color_upper

            # Change trackbar position value to clicked one with tolerance
            cv2.setTrackbarPos('Hue Min', 'Color Calibration', hue_lower)
            cv2.setTrackbarPos('Hue Max', 'Color Calibration', hue_upper)
            cv2.setTrackbarPos('Sat Min', 'Color Calibration', sat_lower)
            cv2.setTrackbarPos('Sat Max', 'Color Calibration', sat_upper)
            cv2.setTrackbarPos('Val Min', 'Color Calibration', val_lower)
            cv2.setTrackbarPos('Val Max', 'Color Calibration', val_upper)

        # --------------------------- SAVE SESSION SECTION -----------------------------
        # Capture a frame from drone camera
        self.frame_read = self.tello.get_frame_read()
        # Take the original frame for the video capture of the session
        frame_original = self.frame_read.frame
        # Check if save session mode is on
        if self.save_session:

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
            self.writer = cv2.VideoWriter("{}/TelloVideo.avi".format(ddir), cv2.VideoWriter_fourcc(*'XVID'),
                                          self.FPS_vid, (width, height))
            self.writer_processed = cv2.VideoWriter("{}/TelloVideo_processed.avi".format(ddir),
                                                    cv2.VideoWriter_fourcc(*'XVID'),
                                                    self.FPS_vid, (width, height))

    def run(self):

        # -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-< CONTINUOUS DRONE CYCLE ->->->->->->->->->->->->->->->->->->->->->

        # --------------------------- SEND DRONE VELOCITY SECTION -----------------------------
        # Function that updates drone velocities in the override mode and autonomous mode
        if self.tello.is_flying:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

        # --------------------------- FRAME READ SECTION -----------------------------
        # If there is no frame, do not read the actual frame
        if self.frame_read.stopped:
            self.frame_read.stop()
            self.drone_continuous_cycle = False

        # Update frame
        frame_original = self.frame_read.frame
        frame = frame_original.copy()

        # --------------------------- DEBUG CALIBRATION SECTION -----------------------------
        # Update the trackbars HSV mask and apply the erosion and dilation
        if self.debug:
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

            lower_hsv = (h_min, s_min, v_min)
            upper_hsv = (h_max, s_max, v_max)

            # Saving the lower and upper hsv values in a pickle file:
            with open('mask_values.pkl', 'wb') as fw:
                pickle.dump([lower_hsv, upper_hsv], fw)
                fw.close()

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
        if self.debug:
            x, y, r, detection, frame_processed = self.object_detection(frame_perspective, lower_hsv, upper_hsv)
        else:
            x, y, r, detection, frame_processed = self.object_detection(frame_perspective, 0, 0)

        # Display grid in the actual frame
        x_1, x_2, y_1, y_2, frame_grid = display_grid(frame_processed, self.grid_size, x, y)

        # Display battery, logo and mode in the video
        frame_user = self.display_icons(self.display_text(frame_grid, 'Drone-x', (410, 25), (0, 0, 0)), bat=True)
        if self.OVERRIDE:
            frame_user = self.display_text(frame_user, 'OVERRIDE MODE: ON', (5, 710), (0, 0, 255), blink=True)
        if self.debug:
            frame_user = self.display_text(frame_user, 'DEBUG MODE: ON', (5, 25), (0, 0, 255), blink=True)

        # --------------------------- READ KEY SECTION -----------------------------
        # Wait for a key to be press and grabs the value
        # k = cv2.waitKey(20)
        # self.set_event(0)
        # time.sleep(0.020)
        k = self.get_event()

        # Press ESC to quit -!-!-!-> EXIT PROGRAM <-!-!-!-
        if k == 27:
            if self.tello.is_flying:
                for i in range(50):
                    self.tello.send_rc_control(0, 0, 0, 0)  # Stop the drone if it has momentum
                    time.sleep(1 / self.FPS)
            self.drone_continuous_cycle = False

        if self.is_taking_off:
            self.tello.get_battery()
            self.tello.takeoff()
            self.is_taking_off = False

        # Press T to take off
        if (k == ord('t') or k == ord('T')) and not self.tello.is_flying and not self.debug:
            self.is_taking_off = True
            print('Takeoff...')
            frame_user = self.display_text(frame_user, 'Taking off...', (5, 25), (0, 255, 255))

        if self.is_landing:
            for i in range(50):
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop the drone if it has momentum
                time.sleep(1 / self.FPS)
            self.tello.land()
            self.is_landing = False

        # Press L to land
        if (k == ord('l') or k == ord('L')) and self.tello.is_flying and not self.debug:
            self.is_landing = True
            print('Land...')
            frame_user = self.display_text(frame_user, 'Landing...', (5, 25), (0, 255, 255))

        # Press spacebar to enter override mode
        if k == 32 and self.tello.is_flying:
            if not self.OVERRIDE:
                self.OVERRIDE = True
                print('OVERRIDE ENABLED')
            else:
                self.OVERRIDE = False
                print('OVERRIDE DISABLED')

        if self.OVERRIDE:
            # W to fly forward and S to fly back
            if k == ord('w') or k == ord('W'):
                self.for_back_velocity = int(self.S * self.oSpeed)
            elif k == ord('s') or k == ord('S'):
                self.for_back_velocity = -int(self.S * self.oSpeed)
            else:
                self.for_back_velocity = 0

            #  C to fly clockwise and Z to fly counter clockwise
            if k == ord('c') or k == ord('C'):
                self.yaw_velocity = int(self.S * self.oSpeed)
            elif k == ord('z') or k == ord('Z'):
                self.yaw_velocity = -int(self.S * self.oSpeed)
            else:
                self.yaw_velocity = 0

            # Q to fly up and E to fly down
            if k == ord('q') or k == ord('Q'):
                self.up_down_velocity = int(self.S * self.oSpeed)
            elif k == ord('e') or k == ord('E'):
                self.up_down_velocity = -int(self.S * self.oSpeed)
            else:
                self.up_down_velocity = 0

            # A to fly left and D to fly right
            if k == ord('d') or k == ord('D'):
                self.left_right_velocity = int(self.S * self.oSpeed)
            elif k == ord('a') or k == ord('A'):
                self.left_right_velocity = -int(self.S * self.oSpeed)
            else:
                self.left_right_velocity = 0

        # --------------------------- AUTONOMOUS SECTION -----------------------------
        if self.tello.is_flying and not self.OVERRIDE and not self.debug:
            # Eliminate pass values
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0

            if detection:
                self.yaw_velocity, self.up_down_velocity, self.for_back_velocity = self.drone_stay_close(x, y, x_1,
                                                                                                         x_2, y_1,
                                                                                                         y_2, r,
                                                                                                         self.radius_stop,
                                                                                                         self.radius_stop_tolerance)

        # --------------------------- WRITE VIDEO SESSION SECTION -----------------------------
        # Save the video session if True
        if self.save_session:
            frame_user = self.display_text(frame_user, 'REC', (810, 25), (0, 0, 0))
            frame_user = self.display_icons(frame_user, bat=True, rec=True)
            self.writer.write(frame_original)
            self.writer_processed.write(frame_user)

        # --------------------------- SHOW VIDEO SECTION -----------------------------
        # Display the video
        cv2.imshow('Drone X', frame_user)

        # --------------------------- MISCELLANEOUS SECTION -----------------------------
        # Save actual time
        self.actual_time = time.time()
        # Delay to showcase desired fps in video
        time.sleep(1 / self.FPS)

    def callback(self, x):
        pass

    def display_icons(self, frame, bat=True, rec=False):
        """ Display icons in the video """

        if bat:
            # Display a battery in the image representing its percentage
            cv2.rectangle(frame, pt1=(920, 5), pt2=(950, 25), color=(255, 255, 255), thickness=2)
            cv2.rectangle(frame, pt1=(950, 9), pt2=(955, 21), color=(255, 255, 255), thickness=2)

            # Request battery every 15 seconds in autonomous mode
            if not self.debug:
                if self.actual_time - self.elapsed_time > 15:
                    self.elapsed_time = self.actual_time
                    print('Solicitar Bateria ')
                    try:
                        self.battery = self.tello.get_battery()  # Get battery level of the drone
                        if not (self.battery == '' or self.battery == 'ok'):  # Checks if string battery is not empty
                            self.battery = int(self.battery)
                            print('Se convirtio valor de bateria a int')
                        else:
                            print('Bateria entrego "" o "ok"')
                    except:
                        print('Error al pedir bateria')


            # Request battery every 24 seconds in debug mode
            elif self.debug:
                if self.actual_time - self.elapsed_time > 24:
                    self.elapsed_time = self.actual_time
                    print('Solicitar Bateria ')
                    try:
                        self.battery = self.tello.get_battery()  # Get battery level of the drone
                        if not (self.battery == '' or self.battery == 'ok'):  # Checks if string battery is not empty
                            self.battery = int(self.battery)
                            print('Se convirtio valor de bateria a int')
                        else:
                            print('Bateria entrego "" o "ok"')
                    except:
                        print('Error al pedir bateria')

            # Display a complete battery
            if self.battery > 75:
                cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 0), thickness=-1)
                cv2.rectangle(frame, pt1=(932, 9), pt2=(938, 21), color=(0, 255, 0), thickness=-1)
                cv2.rectangle(frame, pt1=(940, 9), pt2=(947, 21), color=(0, 255, 0), thickness=-1)
            # Display a 2/3 of the battery
            elif 75 > self.battery > 50:
                cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 255, 255), thickness=-1)
                cv2.rectangle(frame, pt1=(932, 9), pt2=(940, 21), color=(0, 255, 255), thickness=-1)
            # Display 1/3 of the battery
            elif 50 > self.battery > 25:
                cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)
            # Display 1/3 of the battery blinking
            elif self.battery < 25:
                # Blinks battery every 0.5 seconds
                if 0.5 < self.actual_time - self.elapsed_battery_blink < 1:
                    cv2.rectangle(frame, pt1=(924, 9), pt2=(930, 21), color=(0, 0, 255), thickness=-1)
                elif self.actual_time - self.elapsed_battery_blink > 1:
                    self.elapsed_battery_blink = self.actual_time

        if rec:
            # Blinks battery every 1 seconds
            if 1 < self.actual_time - self.elapsed_recording_blink < 2:
                cv2.circle(frame, (890, 15), 10, (0, 0, 255), -1)  # Put a red circle indicating its recording
            elif self.actual_time - self.elapsed_recording_blink > 2:
                self.elapsed_recording_blink = self.actual_time

        return frame

    def display_text(self, frame, text, org, color, blink=False):
        """ Display text in the video """

        font = cv2.FONT_ITALIC
        #  Check if the text is needed to blink
        if blink:
            # This is to make the text blink one second on, one second off
            if 1 < self.actual_time - self.elapsed_text_blink < 2:
                cv2.putText(frame, text=text, org=org, fontFace=font, fontScale=1, color=color,
                            thickness=2, lineType=cv2.LINE_8)
            elif self.actual_time - self.elapsed_text_blink > 2:
                self.elapsed_text_blink = self.actual_time

        #  Display normal text is needed to blink
        else:
            cv2.putText(frame, text=text, org=org, fontFace=font, fontScale=1, color=color,
                        thickness=2, lineType=cv2.LINE_8)

        return frame  # Return the frame with the text

    def object_detection(self, frame, lower_hsv, upper_hsv):
        """ Track the color in the frame """

        # If mode debug is active, make lower and upper parameters
        # be the ones from the trackbars
        if self.debug:
            color_lower = lower_hsv
            color_upper = upper_hsv

        # Else, use the lower and upper hardcoded parameters
        else:
            with open('mask_values.pkl', 'rb') as f:
                color_lower, color_upper = pickle.load(f)
                f.close()

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

    def drone_stay_close(self, x, y, limit_x1, limit_x2, limit_y1, limit_y2, radius, distance_radius, tolerance):
        """
        Control velocities to track object, take x and y for the position of the target,
        limit_x1 to limit_y2 are the lines of the center square of the grid
        r is the radius of the object
        distance_radius is the radius were you want the drone to stop
        tolerance is the range where the drone can keep the distance from the target
        """

        # If the target is in the center, the drone can move back or forward
        if limit_x2 > x > limit_x1 and limit_y2 > y > limit_y1:
            self.for_back_velocity = int((distance_radius - radius) * 1.3333)
            # If the drone is centered with the target and at the distance send velocities to 0
            if distance_radius + tolerance > radius > distance_radius - tolerance:
                self.for_back_velocity = 0
        # Drone move to get the target centered
        else:
            self.yaw_velocity = int((x - 480) * .125)
            self.up_down_velocity = int((300 - y) * .2)
            self.for_back_velocity = 0

        # Send the velocities to drone
        return self.yaw_velocity, self.up_down_velocity, self.for_back_velocity


class Desktop:
    drone = None

    def __init__(self):
        self.drone = DroneX()

    def run(self):
        self.drone.initializer()
        # print('llego')

        while self.drone.drone_continuous_cycle:
            k = cv2.waitKey(20)
            self.drone.set_event(val=k)
            self.drone.run()

        # On exit, print the battery
        self.drone.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()

        # Call it always before finishing. I deallocate resources.
        # self.drone.tello.end()
        if self.drone.tello.is_flying:
            self.drone.tello.land()
        if self.drone.tello.stream_on:
            self.drone.tello.streamoff()

        self.drone.OVERRIDE = False
        print('Se salio del ciclo dron y se termino funcion run')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Drone-X Dashboard")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(270, 20, 261, 101))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap("Resources/Super logo.png"))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")
        self.autonomous_button = QtWidgets.QPushButton(self.centralwidget)
        self.autonomous_button.setGeometry(QtCore.QRect(310, 200, 191, 71))
        self.autonomous_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
                                             "color: rgb(255, 255, 255);\n"
                                             "font: 12pt \"Impact\";")
        self.autonomous_button.setObjectName("autonomous_button")
        self.calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.calib_button.setGeometry(QtCore.QRect(310, 330, 191, 71))
        self.calib_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
                                        "color: rgb(255, 255, 255);\n"
                                        "font: 12pt \"Impact\";")
        self.calib_button.setObjectName("calib_button")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(310, 460, 191, 71))
        self.exit_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
                                       "color: rgb(255, 255, 255);\n"
                                       "font: 12pt \"Impact\";")
        self.exit_button.setObjectName("exit_button")
        self.about_button = QtWidgets.QPushButton(self.centralwidget)
        self.about_button.setGeometry(QtCore.QRect(670, 500, 111, 31))
        self.about_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
                                        "color: rgb(255, 255, 255);\n"
                                        "font: 8pt \"Impact\";")
        self.about_button.setObjectName("about_button")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(310, 120, 211, 71))
        self.radioButton.setStyleSheet("font: 10pt \"Impact\";")
        self.radioButton.setObjectName("radioButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.desktop = Desktop()
        self.radioButton.toggled.connect(self.toggle_session)
        self.calib_button.clicked.connect(self.debug_mode)
        self.autonomous_button.clicked.connect(self.autonomous_mode)
        self.exit_button.clicked.connect(self.close_gui)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Drone-X Dashboard"))
        self.autonomous_button.setStatusTip(
            _translate("MainWindow", "Entrar en vuelo autónomo que seguira el objeto a detectar"))
        self.autonomous_button.setText(_translate("MainWindow", "Modo Autónomo"))
        self.calib_button.setStatusTip(
            _translate("MainWindow", "Entrar en modo de calibración para ajustar valores a detectar objeto deseado"))
        self.calib_button.setText(_translate("MainWindow", "Calibración"))
        self.exit_button.setStatusTip(_translate("MainWindow", "Salir del programa"))
        self.exit_button.setText(_translate("MainWindow", "Salir"))
        self.about_button.setStatusTip(_translate("MainWindow", "Documentación del programa"))
        self.about_button.setText(_translate("MainWindow", "Mas Info"))
        self.radioButton.setText(_translate("MainWindow", "Guardar video de sesión"))

    def toggle_session(self):
        # print('llego')
        self.desktop.drone.save_session = self.radioButton.isChecked()
        print(self.desktop.drone.save_session)
        # Desktop.drone.save_session = self.radioButton

    def debug_mode(self):
        if not self.desktop.drone.drone_continuous_cycle:
            self.desktop.drone.debug = True
            try_error = False
            try:
                self.desktop.drone.tello.retry_count = 1
                self.connecting_popup()
                self.desktop.drone.tello.send_control_command('command', timeout=3)
            except:
                try_error = True
                print("Tello not connected")
                self.not_connected_popup()

            if not try_error:
                self.desktop.drone.tello.retry_count = 3
                self.desktop.run()

        else:
            print('Ya esta en un modo')

    def autonomous_mode(self):
        if not self.desktop.drone.drone_continuous_cycle:
            self.desktop.drone.debug = False
            try_error = False
            try:
                self.desktop.drone.tello.retry_count = 1
                self.connecting_popup()
                self.desktop.drone.tello.send_control_command('command', timeout=3)
            except:
                try_error = True
                print("Tello not connected")
                self.not_connected_popup()

            if not try_error:
                self.desktop.drone.tello.retry_count = 3
                self.desktop.run()
        else:
            print('Ya esta en un modo')

    def not_connected_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText('Tello no conectado a Wifi')
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setInformativeText('Favor de conectarse al Tello, y volver a intentar')
        x = msg.exec_()

    def connecting_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle('Conectandose')
        msg.setText('El Tello se esta conectando')
        msg.setIcon(QMessageBox.Information)
        msg.setInformativeText('Favor de ser paciente')
        x = msg.exec_()

    def close_gui(self):
        if self.desktop.drone.tello.background_frame_read is not None:
            self.desktop.drone.tello.background_frame_read.stop()
        if self.desktop.drone.tello.cap is not None:
            self.desktop.drone.tello.cap.release()
        self.close()


def main():
    # drone = Desktop()
    #
    # # Run Drone-X
    # drone.run()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
