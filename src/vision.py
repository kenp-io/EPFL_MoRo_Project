# ******** IMPORTS ********

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import queue
import threading

# ******** CLASSES ********

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
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
    return self.q.get()

# class for getting values in images
class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        return 'x={:.01f}, y={:.01f}, val = '.format(x, y)

# ******** FUNCTIONS ********

def printImageValues(image):
    get_ipython().run_line_magic('matplotlib', 'widget')
    fig, ax = plt.subplots()
    im = ax.imshow(image, interpolation='none')
    ax.format_coord = Formatter(im)
    plt.show()
    return

def find_thymio_center(frame):
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take green channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    #printImageValues(hsv)
    mask_green = cv2.inRange(hsv, (60, 60, 60), (80, 255,255))
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
    circles_green = cv2.HoughCircles(green_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=70, param2 = 10,
                                     minRadius = 0, maxRadius = 50)
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
    #ERROR HERE IF NO CIRCLES - NEED FIXING
    robot_center_absolute = [circles_green[0][0], 720 - circles_green[0][1]]
    return robot_center_absolute, output_green

def find_thymio_front(frame):
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take blue channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    #printImageValues(hsv)
    mask_blue = cv2.inRange(hsv, (110, 60, 60), (130, 255,255))
    ## slice the blue
    imask = mask_blue>0
    blue = np.zeros_like(frame, np.uint8)
    blue[imask] = blurred_frame[imask]
    # Apply hough transform on blue image
    # Convert image to grayscale
    blue_gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    circles_blue = cv2.HoughCircles(blue_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2 = 10,minRadius = 0, maxRadius = 60)
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
    #ERROR HERE IF NO CIRCLES - NEED FIXING
    robot_front_absolute = [circles_blue[0][0], 720 - circles_blue[0][1]]
    return robot_front_absolute, output_blue

def find_destination_center(frame):
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.5)
    # Convert image to HSV and only take red channel
    ## convert to hsv
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (-10, 100, 100), (10, 255,255)) + \
    cv2.inRange(hsv, (170, 100, 100), (190, 255, 255))
    ## slice the red
    imask = mask_red>0
    red = np.zeros_like(frame, np.uint8)
    red[imask] = blurred_frame[imask]
    # Apply hough transform on red image
    # Convert image to grayscale
    red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    circles_red = cv2.HoughCircles(red_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2 = 20, minRadius = 30, maxRadius = 150)
    # ensure at least some circles were found
    output_destination = frame.copy()
        #print(circles_red)
    if circles_red is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_red = np.round(circles_red[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_red:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output_destination, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output_destination, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
    #ERROR HERE IF NO CIRCLES - NEED FIXING
    destination_center_absolute = [circles_red[0][0], 720 - circles_red[0][1]]
    return destination_center_absolute, output_destination

def find_objects(frame_objects):
    blurred_objects = cv2.GaussianBlur(frame_objects, (5, 5), 1.5)
    #edges_objects = cv2.Canny(blurred_objects, 50, 100)
    #Only take absolute black objects :
    hsv = cv2.cvtColor(frame_objects, cv2.COLOR_BGR2HSV)
        #printImageValues(hsv)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask_black = cv2.inRange(hsv, (0, 0, 0), (170, 255,150))
        #get_ipython().run_line_magic('matplotlib', 'inline')
        #plt.imshow(mask_black)
        #plt.show()
    ## slice the black
    imask = mask_black>0
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
    contours, hierarchy = cv2.findContours(gray_objects,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output_objects,[box],0,(255,255,255),150)
    return output_objects

# ******** NOT SURE IF NEED THOSE FUNCTIONS ********

def mouseHSV(event,x,y,flags,param):
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
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('HSV')
    cv2.setMouseCallback('HSV',mouseHSV)
    while(1):
        cv2.imshow('HSV',hsv)
        if cv2.waitKey(0):
            break
    #if esc pressed, finish.
    cv2.destroyAllWindows()
