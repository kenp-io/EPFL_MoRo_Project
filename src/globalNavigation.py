"""
## globalNavigation.py ##

Functions used in the global navigation algorithm.

Imported by : main.ipynb, localNavigation.py, utils.py, vision.py.

*** Functions ***
variable_info(variable)
create_empty_plot(max_val)
_get_movements_4n()
_get_movements_8n()
reconstruct_path(cameFrom, current)
A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", max_val = MAX_VAL)
runAstar(start, goal, max_val, occupancy_grid, cmap)
display_occupancy_grid(output_objects)
restructurePath(path)
upscalePath(path)
angleDifference(angleRef, angleGoal)
angleCalculatorPath(robotFront, robotCenter, destinationCenter)
distanceCalculator(current, goal)
angleTwoPoints(pointGoal, pointStart)
turnAngle(angle, ourThymio)
goForward(distance, ourThymio)
stopForward(ourThymio)
getAbsoluteAngle(pointA, pointB)
pathSimplifier(path)
followPath(ourThymio, path)

"""

# ******** IMPORTS ********

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import colors
from Thymio import Thymio
import threading
import time

import utils
import kalman
import vision
import localNavigation

# ******** CONSTANTS ********

RATIO      = 16/9     # wanted ratio for the image to analyse
DOWNSIZE   = 0.0694   # ratio to reduce the number of pixel
MAX_VAL    = 50       # map size
PATH_START = 0        # index of path if beginning of new path (first step)

# Simplifier constants
SIMP_THRESHOLD = 1
SIMP_THRESHOLD_CLOSE = 5

# ******** FUNCTIONS ********

def create_empty_plot(max_val):
    """
    Helper function to create a figure of the desired dimensions & grid

    :param max_val: dimension of the map along the x and y dimensions

    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    major_ticks_x = np.arange(0, max_val*RATIO+1, 5)
    minor_ticks_x = np.arange(0, max_val*RATIO+1, 1)
    major_ticks_y = np.arange(0, max_val+1, 5)
    minor_ticks_y = np.arange(0, max_val+1, 1)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val])
    ax.set_xlim([-1,max_val*RATIO])
    ax.grid(True)

    return fig, ax

def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).

    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]

def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).

    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = np.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]

def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node
                     immediately preceding it on the cheapest path from start
                     to n currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current])
        current=cameFrom[current]
    return total_path

def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", max_val = MAX_VAL):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    ##:param h:
    ##:param coords:
    :param occupancy_grid: the grid map
    :param movement_type: select between 4-connectivity ('4N') and
                          8-connectivity ('8N', default)
    :param max_val: size of the analysed window in pixel

    :return: a tuple that contains: (the resulting path in meters, the
             resulting path in data array indices)
    """

    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------

    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        assert point[0]>=0 and point[0]<max_val*RATIO+1, "start or end goal not contained in the map, x-axis"
        assert point[1]>=0 and point[1]<max_val, "start or end goal not contained in the map, y-axis"

    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')

    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------

    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]

    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:

        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet

        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)

        #for each neighbor of current:
        for dx, dy, deltacost in movements:

            neighbor = (current[0]+dx, current[1]+dy)

            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue

            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet):
                continue

            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost

            if neighbor not in openSet:
                openSet.append(neighbor)

            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet

def runAstar(start, goal, max_val, occupancy_grid, cmap):
    """
    Run A* algorithm to go from start to goal and simplifies the computed path
    to reduce the number of checkpoint the thymio has to visit, thus avoiding
    jerky behavior while following the global path

    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param max_val: max amount of pixel wanted on y axis (size of image)
    :param occupancy_grid: the grid map
    :param cmap: colormap for matplotlib plot functions

    :return: simplePathUpscaled is the path found by A*, simplified and rescaled
    """


    x,y = np.mgrid[0:max_val*RATIO+1:1, 0:max_val:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run the A* algorithm
    path, visitedNodes = A_Star(start, goal, h, coords, cv2.flip(occupancy_grid, 0).transpose(),
                                movement_type="8N")
    path = np.array(path).reshape(-1, 2).transpose()
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()

    # Puts the path in the form of coordinates: [(x1,y1),(x2,y2),..] instead of two lists of axis values
    pathRestructed = restructurePath(path)
    print('Path Original:')
    print(pathRestructed)
    # Simplifies the number of checkpoint that the thymio has to visit
    simplePath = pathSimplifier(pathRestructed)
    # Rescales the path in 1280:720p
    simplePathUpscaled = upscalePath(simplePath)
    print('Path Simple:')
    print(simplePathUpscaled)


    # Displaying the map
    fig_astar, ax_astar = create_empty_plot(max_val)
    ax_astar.imshow(cv2.flip(occupancy_grid, 0), cmap=cmap)

    # Plot the best path found and the list of visited nodes
    #ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange');
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);
    ax_astar.scatter([i[0] for i in simplePath], [i[1] for i in simplePath], marker="o", color = 'red', s=200);
    return simplePathUpscaled

def display_occupancy_grid(output_objects):
    """
    Displays the grid map of the environnement, with expanded objects

    ##:param output_object:

    :return: the grid map and the colormap
    """
    #img with expanded objects

    test = output_objects.copy()
    compressed = cv2.resize(test, (0, 0), fx = DOWNSIZE, fy = DOWNSIZE) #coef to resize img in less pixels
    #plt.imshow(compressed)
    #Creating the grid
    fig, ax = create_empty_plot(MAX_VAL) # MAX_VAL Size of the map
    # Creating the occupancy grid
    data = compressed.copy() # Create a grid of 67 x 50 with objects values
    #print(data)
    cmap = colors.ListedColormap(['white', 'black']) # Select the colors with which to display objects and free cells
    # Converting the random values into occupied and free cells
    limit = 10
    occupancy_grid = data.copy()
    #occupancy_grid_flipped = cv2.flip(occupancy_grid, flipCode = 0)
    occupancy_grid[data>limit] = 1
    occupancy_grid[data<=limit] = 0
    # Displaying the map
    ax.imshow(cv2.flip(occupancy_grid, 0))
    plt.title("Map : free cells in white, occupied cells in black");
    return occupancy_grid, cmap

def restructurePath(path):
    """
    Restructures the path format from lists of coordinates along the same axis
    ([(x,x,x),(y,y,y)]) to a list of points coordinates [(x,y),(x,y),...]

    :param path: path computed by the A* algorithm in the format [(x1,x2,x3),(y1,y2,y3)]

    :return: finalPath is the path in a point coordinates format [(x1,y1),(x2,y2),...]
    """

    finalPath = []
    for i in range(len(path[0])):
        finalPath.append([path[0][i],path[1][i]])
    return finalPath

def upscalePath(path):
    """
    Resizes the path coordinates to the real initial image size

    :param path: path computed by the A* algorithm, simplified and
                 in the format [(x1,y1),(x2,y2),...]

    :return: path upscaled to real size (1280x720)
    """
    #upscale the path
    new_path = np.asarray(path)*(1/DOWNSIZE)
    new_path = np.round(new_path, 0)
    new_path = new_path.astype(int)
    #new_path = [[round(x[0]) for x in new_path], [round(y[0]) for y in new_path]]
    return new_path

def angleDifference(angleRef, angleGoal):
    """
    Computes an angle difference between two "position angle" of the Thymio.
    Those angles are referenced the same way has in the unit trigonometric circle,
    with the camera x axis aligned with the circle axis.

    :param angleRef: reference (initial) angle of the Thymio compared to the
                     unit vector of the x-axis
    :param angleGoal: goal "position angle" of the Thymio

    :return: angular distance to reach angleGoal starting from angleRef
    """
        #print(f'angleRef: {np.rad2deg(angleRef)}')
        #print(f'angleGoal: {np.rad2deg(angleGoal)}')
    angleToTurn = (angleGoal - angleRef)%(2*np.pi)
    if angleToTurn > np.pi:
        return angleToTurn-(2*np.pi)
    else:
        return angleToTurn

def angleCalculatorPath(robotFront, robotCenter, destinationCenter):
    """
    Computes the angular distance to travel to orient the Thymio in the destinationCenter direction

    :param robotFront: position(coordinates) of the blue circle center at the front of Thymio
    :param: robotcenter: position of the center green circle center at the center of the Thymio
    :param: destinationCenter: position of the next destination point

    :return: angular distance to turn the Thymio in the right direction
    """
    angleRobot = angleTwoPoints(robotFront, robotCenter)
        #print("Robot : ", np.rad2deg(angleRobot))
    angleGoal = angleTwoPoints(destinationCenter, robotFront)
        #print("Angle goal absolute:", np.rad2deg(angleGoal))
    angleToTurn = angleDifference(angleRobot, angleGoal)
    return angleToTurn

def distanceCalculator(current, goal):
    """
    Computes the absolute distance between two coordinate points

    :param current: current position in (x,y) coordinates
    :param goal: goal position in (x,y) coordinates

    :return: distances between the two points
    """
    return np.sqrt((goal[0]-current[0])**2+(goal[1]-current[1])**2)

def angleTwoPoints(pointGoal, pointStart):
    """
    Computes the absolute angle of the vector from start and goal points.
    Those angles are referenced the same way has in the unit trigonometric circle,
    with the camera x axis aligned with the circle axis.

    :param pointGoal: end extremity coordinates of the vector
    :param: pointStart: origin coordinates of the vector

    :return: absolut angle of the vector
    """
    angleRobotAbsolute = np.arctan2(pointGoal[1] - pointStart[1], pointGoal[0] - pointStart[0])
    return angleRobotAbsolute

def turnAngle(angle, ourThymio):
    """
    Make the Thymio turn on himself for a certain angle

    :param angle: rotation angle wanted
    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods
    """
    sleepTime = utils.FULLROTATIONTIME/(2*np.pi)*abs(angle)/1000
    if angle > 0:
        ourThymio.antiClockwise()
        time.sleep(sleepTime)
        ourThymio.stop()
    elif angle < 0:
        ourThymio.clockwise()
        time.sleep(sleepTime)
        ourThymio.stop()

def goForward(distance, ourThymio):
    """
    Activates Thymio's motors to go forward and launch a thread timer, which
    will stop the robot after a adapted time (using stopForward function)
    to make the Thymio travel a certain distance.

    :param distance: distance to travel forward in mm
    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods
    """
    sleepTime = distance/utils.FORWARDCONSTANT
    ourThymio.forward()
    t = threading.Timer(sleepTime, lambda: stopForward(ourThymio))
    t.start()

def stopForward(ourThymio):
    """
    Stop the Thymio's motors after a timer is finished.
    If the robot has entered the local avoidance mode it won't stop the motors.

    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods
    """
    ourThymio.reached = True
    if not ourThymio.inLocal:
        ourThymio.stop()
    ourThymio.stopKalmanFlag.set()

def getAbsoluteAngle(pointA, pointB):
    """
    Compute the algebraic slope of the vector A->B, with 100 and -100 to
    imitate positive and negative infinity (if both points have the same x coordinate)

    :param pointA: coordinates point
    :param pointB: coordinates point

    :return: slope value of vector A->B (± 100 for infinity)
    """
        #print(f'point A: {pointA}')
        #print(f'point B: {pointB}')
    if (pointB[0]-pointA[0]) == 0:
        if (pointB[1]-pointA[1])>0:
            return 100
        else:
            return -100
    return (pointB[1]-pointA[1])/(pointB[0]-pointA[0])

def pathSimplifier(path):
    """
    Simplifies the complex path computed by A* algorithm into a list of
    checkpoints that the Thymio will have to visit to avoid obstacles

    :param path: path computed by the A* algorithm in the format [(x1,y1),(x2,y2),...]

    :return: list of checkpoint coordinates corresponding to the simplified path
    """

    index = 0
    simplePath = []
    simplePath.append(path[index])
    index = index + 1
    finalIndex = index
    while finalIndex < len(path)-1: #loops in paths points
            #print(f'Main loop')
            #print(f'Main index: {index}')
            #print(f'Main finalIndex: {finalIndex}')
        lookingFurther = 0
        index = finalIndex
        refAngle = getAbsoluteAngle(path[index],path[index+1])
        while index < len(path)-1:
                #print(f'Inside loop')
                #print(f'index:{index}')
            newAngle = getAbsoluteAngle(path[index], path[index+1])
            #print(abs(refAngle-newAngle))
            if abs(refAngle-newAngle) > 0.2:
                if lookingFurther < SIMP_THRESHOLD:
                    lookingFurther = lookingFurther+1
                else:
                    break
            else:
                lookingFurther = 0
                finalIndex = index+1
                if finalIndex > len(path)-1-SIMP_THRESHOLD_CLOSE:
                    finalIndex = len(path)-1
                    break
            index = index + 1
                #print(f'Inside index: {index}')
                #print(f'Inside finalIndex: {finalIndex}')
        simplePath.append(path[finalIndex])

    return simplePath

def followPath(ourThymio, path):
    """
    Function called in main loop to control the behaviour of the Thymio in order
    to follow the computed simplified and upscaled path. Called only once if no
    object is locally detected and it loops until all checkpoints are reached
    by Thymio.

    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods
    :param path: path computed by the A* algorithm, simplified and in the format
                 [(x1,y1),(x2,y2),...]

    :return: boolean indicating if Thymio reached the destination, return False
             if Thymio has entered local avoidance mode
    """

    for index in range(len(path)-1):

        print(f'Point index: {index}')

        #Turns to face the next goal in path
        if index == PATH_START:
            angleToTurn = angleDifference(ourThymio.angle, angleTwoPoints(path[index+1],
                                                                          ourThymio.getCenter()))
            print(f'angleToTurn: {np.rad2deg(angleToTurn)}')
            turnAngle(angleToTurn, ourThymio)
        else:
            angleToTurn = angleCalculatorPath(path[index],path[index-1],path[index+1])
            print(f'angleToTurn: {np.rad2deg(angleToTurn)}')
            turnAngle(angleToTurn, ourThymio)

        #Goes forward towards next goal, kalman and local avoidance is active
        ourThymio.reached = False
        ourThymio.clearKalman()
        kThread = kalman.kalmanThread(ourThymio)

        distance = distanceCalculator(path[index], path[index+1])
        goForward(distance, ourThymio)
        kThread.start()

        while not ourThymio.reached:
            #check if collision
            wentInLocal = localNavigation.localCheck(ourThymio)
            #check if local nav ended
            if wentInLocal:
                return False
            # kalman is executed automatically every DT in kThread
            time.sleep(0.1)

    return True
