"""
## vision.py ##

Definition of constants, classes and functions used in the computer vision algorithm for detection.

Imported by: main.ipynb, globalNavigation.py, utils.py.

*** Classes ***
VideoCapture(name)
Formatter(image)

*** Functions ***
printImageValues(image)
find_thymio_center(frame)
find_thymio_front(frame)
find_destination_center(frame)
find_objects(frame_objects)
mouseHSV(event,x,y,flags,param)
mouseRGB(event,x,y,flags,param)
display_hsv(frame)
"""

# ******** IMPORTS ********

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import queue
import threading

# ******** CONSTANTS ********

HGREEN_LOW     = 60
HBLUE_LOW      = 110
HRED_LOW1      = -10
HRED_LOW2      = 170
HBLACK_LOW     = 0
HGREEN_HIGH    = 80
HBLUE_HIGH     = 130
HRED_HIGH1     = 10
HRED_HIGH2     = 190
HBLACK_HIGH    = 170
SAT_LOW        = 60
SATRED_LOW     = 100
SATBLACK_LOW   = 0
SAT_HIGH       = 255
VAL_LOW        = 60
VALRED_LOW     = 100
VALBLACK_LOW   = 0
VAL_HIGH       = 255
VALBLACK_HIGH = 150
CAMERA_Y_SIZE = 720

GREEN_RADMIN   = 0
GREEN_RADMAX   = 50
BLUE_RADMIN    = 0
BLUE_RADMAX    = 60
RED_RADMIN     = 30
RED_RADMAX     = 150

#Hough parameters
MIN_CIRCLE_DIST = 100
DP              = 1.2

# ******** CLASSES ********

# bufferless VideoCapture
class VideoCapture:
    """
    Class gathering initialisation and reading methods to obtain frames from the camera

    **Initialisation**
    VideoCapture(name)

    **Functions**
    _reader()
    read()
    """

    def __init__(self, name):
        """
        Get a camera capture using open cv2 library and start a thread
        to read the captures

        :param name: name of the camera in our system (e.g. '0' for front
                     computer camera, or IP address if using a smartphone
                     application,...)
        """
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        """
        Reads frames as soon as they are available, keeping only most recent one
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """
        Gives the frame in the self.q queue

        :return: frame captured with the camera
        """
        return self.q.get()

class Formatter(object):
    """
    Class for getting values in captured images

    **Initialisation**
    Formatter(im)

    **Functions**
    ___call__(x,y)
    """

    def __init__(self, im):
        """
        Initialises the object with the images im

        :param im: image that we want to get the values
        :return: creates the object of class Formatter
        """
        self.im = im

    def __call__(self, x, y):
        """
        Used to show pixel values when the cursor is on top of it on the imgee

        :param x: position x of pixel
        :param y: position y of pixel

        :return: position and value
        """
        return 'x={:.01f}, y={:.01f}, val = '.format(x, y)

# ******** FUNCTIONS ********

def printImageValues(image):
    """
    ## Shows matplotlib plot

    :param image: image that we want to get and show the values
    """

    get_ipython().run_line_magic('matplotlib', 'widget')
    fig, ax = plt.subplots()
    im = ax.imshow(image, interpolation='none')
    ax.format_coord = Formatter(im)
    plt.show()
    return

def find_thymio_center(frame):
    """
    If present on the frame, finds the Thymio center green
    circle using open cv2 library and hsv color format

    :param frame: Image of the environnement obtained from the camera in format 1280x720

    :return: if detected, returns the position of Thymio's center and the
             frame with the green circle identified
    """
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take green channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        #printImageValues(hsv)
    mask_green = cv2.inRange(hsv, (HGREEN_LOW, SAT_LOW, VAL_LOW), (HGREEN_HIGH, SAT_HIGH,VAL_HIGH))
        #get_ipython().run_line_magic('matplotlib', 'inline')
        #plt.imshow(mask_green)
        #plt.show()
    ## slice the green
    imask = mask_green>0
    green = np.zeros_like(frame, np.uint8)
    green[imask] = blurred_frame[imask]
    # Apply hough transform on green image
    # Convert image to grayscale
    green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    circles_green = cv2.HoughCircles(green_gray, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_CIRCLE_DIST,
                                     param1=70, param2=10, minRadius=GREEN_RADMIN,
                                     maxRadius=GREEN_RADMAX)
    # ensure at least some circles were found
    output_green = frame.copy()
    if circles_green is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_green = np.round(circles_green[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_green:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output_green, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output_green, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        robot_center_absolute = [circles_green[0][0], CAMERA_Y_SIZE - circles_green[0][1]]
        return robot_center_absolute, output_green
    else:
        return None, None

def find_thymio_front(frame):
    """
    If present on the frame, finds the Thymio front blue circle using
    open cv2 library and hsv color format

    :param frame: Image of the environnement obtained from the camera in format 1280x720

    :return: if detected, returns the position Thymio's front and the frame
             with the blue circle identified
    """

    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take blue channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, (HBLUE_LOW, SAT_LOW, VAL_LOW), (HBLUE_HIGH, SAT_HIGH, VAL_HIGH))
    ## slice the blue
    imask = mask_blue>0
    blue = np.zeros_like(frame, np.uint8)
    blue[imask] = blurred_frame[imask]
    # Apply hough transform on blue image
    # Convert image to grayscale
    blue_gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    circles_blue = cv2.HoughCircles(blue_gray, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_CIRCLE_DIST,
                                    param1=100, param2=10, minRadius=BLUE_RADMIN,
                                    maxRadius=BLUE_RADMAX)
    output_blue = frame.copy()
    # ensure at least some circles were found
    if circles_blue is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_blue = np.round(circles_blue[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_blue:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output_blue, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output_blue, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        robot_front_absolute = [circles_blue[0][0], CAMERA_Y_SIZE - circles_blue[0][1]]
        return robot_front_absolute, output_blue
    else:
        return None, None

def find_destination_center(frame):
    """
    If present on the frame, finds destination red circle using open cv2 library and hsv color format

    :param frame: Image of the environnement obtained from the camera in format 1280x720

    :return: if detected, returns center of the destination red circle and the
             frame with the red circle identified
    """
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take red channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (HRED_LOW1, SATRED_LOW, VALRED_LOW), (HRED_HIGH1, SAT_HIGH,VAL_HIGH)) + \
               cv2.inRange(hsv, (HRED_LOW2, SATRED_LOW, VALRED_LOW), (HRED_HIGH2, SAT_HIGH,VAL_HIGH))
    ## slice the red
    imask = mask_red>0
    red = np.zeros_like(frame, np.uint8)
    red[imask] = blurred_frame[imask]
    # Apply hough transform on red image
    # Convert image to grayscale
    red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    circles_red = cv2.HoughCircles(red_gray, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_CIRCLE_DIST,
                                   param1=100, param2=20, minRadius=RED_RADMIN,
                                   maxRadius=RED_RADMAX)
    # ensure at least some circles were found
    output_destination = frame.copy()

    if circles_red is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_red = np.round(circles_red[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_red:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output_destination, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output_destination, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        destination_center_absolute = [circles_red[0][0], CAMERA_Y_SIZE - circles_red[0][1]]
        return destination_center_absolute, output_destination
    else:
        return None, None

def scale_contour(cnt, scale):
    """
    Rescale the contour of the objects to expand their boundaries
    """

    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def find_objects(frame_objects):
    """
    If present on the frame, finds the objects of our environment using open cv2 and hsv color format

    :param frame_object: Image of the environnement obtained from the camera in format 1280x720

    :return: if objects are detected, returns a np array of the size of frame with detected objects
    """
    blurred_objects = cv2.GaussianBlur(frame_objects, (5, 5), 1.5)
    #Only take absolute black objects :
    hsv = cv2.cvtColor(frame_objects, cv2.COLOR_BGR2HSV)
        #printImageValues(hsv)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask_black = cv2.inRange(hsv, (HBLACK_LOW, SATBLACK_LOW, VALBLACK_LOW),
                             (HBLACK_HIGH, SAT_HIGH,VALBLACK_HIGH))
        #get_ipython().run_line_magic('matplotlib', 'inline')
        #plt.imshow(mask_black)
        #plt.show()
    ## slice the black
    imask = mask_black > 0
    black = np.zeros_like(frame_objects, np.uint8)
    black[imask] = 255
    gray_objects = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        #plt.figure(figsize = (50,10))
        #plt.imshow(gray_objects)
        #plt.show()
    output_objects = np.zeros_like(black)
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(gray_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(black, contours, -1, (0, 0, 255), -1)
        #plt.figure(figsize = (50,10))
        #plt.imshow(black)
        #plt.show()
    # Draw all contours
    # -1 signifies drawing all contours
    for cnt in contours:

        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            #print(area, len(approx))
        if (area > 7000) & (area < 13000):
            if(len(approx) in range(4, 6)):
                cnt = scale_contour(cnt, 2.9)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output_objects,[box],0,(255,255,255),-1)
    plt.imshow(output_objects)
    return output_objects

# ******** Troubleshooting functions ********

def mouseHSV(event,x,y,flags,param):
    """
    Prints HSV values of the pixel on which the mouse has clicked

    :param event: type of event (here we test if it is a Left mouse Button Down)
    :param x: x coordinates with opencv axis orientation
    :param y: y coordinates with opencv axis orientation
    :param flags: potential usefull flags
    :param param: potential usefull parameters
    """
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        H = hsv[y,x,0]
        S = hsv[y,x,1]
        V = hsv[y,x,2]
        values = hsv[y,x]
        print("H: ",H)
        print("S: ",S)
        print("V: ",V)
        print("HSV Format: ",values)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

def mouseRGB(event,x,y,flags,param):
    """
    Prints RGB values of the pixel on which the mouse has clicked

    :param event: type of event (here we test if it is a Left mouse Button Down)
    :param x: x coordinates with opencv axis orientation
    :param y: y coordinates with opencv axis orientation
    :param flags: potential usefull flags
    :param param: potential usefull parameters
    """

    if (event == cv2_EVENT_LBUTTONDOWN):
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BGR Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

def display_hsv(frame):
    """
    Displays hsv frame

    :param frame: Image of the environnement obtained from the camera in format 1280:720p
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('HSV')
    cv2.setMouseCallback('HSV',mouseHSV)
    while(1):
        cv2.imshow('HSV',hsv)
        if cv2.waitKey(0):
            break
    #if esc pressed, finish.
    cv2.destroyAllWindows()
