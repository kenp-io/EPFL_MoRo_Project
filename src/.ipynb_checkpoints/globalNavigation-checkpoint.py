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
#maybe to remove
import vision

REACHED = 0

# ******** FUNCTIONS ********

def variable_info(variable):
    """
    Provided a variable, prints the type and content of the variable
    """
    print("This variable is a {}".format(type(variable)))
    if type(variable) == np.ndarray:
        print("\n\nThe shape is {}".format(variable.shape))
    print("\n\nThe data contained in the variable is : ")
    print(variable)
    print("\n\nThe elements that can be accessed in the variable are :\n")
    print(dir(variable))

def create_empty_plot(max_val):
    """
    Helper function to create a figure of the desired dimensions & grid

    :param max_val: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    RATIO = 16/9
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
    :param cameFrom: map (dictionary) containing for each node n the node immediately
                     preceding it on the cheapest path from start to n
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current])
        current=cameFrom[current]
    return total_path

def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", max_val = 50):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """

    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------
    RATIO = 16/9
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
    RATIO = 16/9
    x,y = np.mgrid[0:max_val*RATIO+1:1, 0:max_val:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run the A* algorithm
    path, visitedNodes = A_Star(start, goal, h, coords, cv2.flip(occupancy_grid, 0).transpose(), movement_type="8N")
    path = np.array(path).reshape(-1, 2).transpose()
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()

    # Displaying the map
    fig_astar, ax_astar = create_empty_plot(max_val)
    ax_astar.imshow(cv2.flip(occupancy_grid, 0), cmap=cmap)

    # Plot the best path found and the list of visited nodes
    #ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange');
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);
    return path

def display_occupancy_grid(output_objects):
    test = output_objects.copy()
    compressed = cv2.resize(test, (0, 0), fx = 0.0694, fy = 0.0694)
    #plt.imshow(compressed)
    #Creating the grid
    max_val = 50 # Size of the map
    fig, ax = create_empty_plot(max_val)
    # Creating the occupancy grid
    data = compressed.copy() # Create a grid of 67 x 50 with objects values
    #print(data)
    cmap = colors.ListedColormap(['white', 'black']) # Select the colors with which to display obstacles and free cells
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

def transformPath(path):
    #upscale the path
    new_path = path*14.41
    new_path = [[round(x) for x in new_path[0]], [round(y) for y in new_path[1]]]
    #transform its shape
    finalPath = []
    for i in range(len(path[0])):
        finalPath.append([new_path[0][i],new_path[1][i]])
    return finalPath

def angleDifference(angleRef, angleGoal):
        #print(f'angleRef: {np.rad2deg(angleRef)}')
        #print(f'angleGoal: {np.rad2deg(angleGoal)}')
    angleToTurn = (angleGoal - angleRef)%(2*np.pi)
    if angleToTurn > np.pi:
        return angleToTurn-(2*np.pi)
    else:
        return angleToTurn

def angleCalculatorPath(robot_front_absolute, robot_center_absolute, destination_center_absolute):
    angleRobotAbsolute = np.arctan2(robot_front_absolute[1] - robot_center_absolute[1],
                                    robot_front_absolute[0] - robot_center_absolute[0])
        #print("Robot : ", np.rad2deg(angleRobotAbsolute))
    angleGoalAbsolute = np.arctan2(destination_center_absolute[1] - robot_front_absolute[1],
                                   destination_center_absolute[0] - robot_front_absolute[0])
        #print("Angle goal absolute:", np.rad2deg(angleGoalAbsolute))
    angleToTurn = np.rad2deg(angleGoalAbsolute - angleRobotAbsolute)%360
    if angleToTurn > 180:
        return angleToTurn-360
    else:
        return angleToTurn

def distanceCalculator(current, goal):
    return np.sqrt((goal[0]-current[0])**2+(goal[1]-current[1])**2)

def angleTwoPoints(pointGoal, pointStart):
    angleRobotAbsolute = np.arctan2(pointGoal[1] - pointStart[1], pointGoal[0] - pointStart[0])
    return angleRobotAbsolute

def turnAngle(angle, ourThymio):
    sleepTime = utils.FULLROTATIONTIME/(2*np.pi)*abs(angle)/1000
    if angle > 0:
        ourThymio.clockwise()
        time.sleep(sleepTime)
        ourThymio.stop()
    elif angle < 0:
        ourThymio.antiClockwise()
        time.sleep(sleepTime)
        ourThymio.stop()

def goForward(distance, ourThymio):
    sleepTime = distance/utils.FORWARDCONSTANT
    ourThymio.forward()
    t = threading.Timer(sleepTime, stopForward, [ourThymio])
    t.start()

def stopForward(ourThymio):
    global IN_LOCAL
    global REACHED
    
    if not IN_LOCAL:
        ourThymio.stop()
    REACHED = True

def getAbsoluteAngle(pointA, pointB):
        #print(f'point A: {pointA}')
        #print(f'point B: {pointB}')
    if (pointB[0]-pointA[0]) == 0:
        if (pointB[1]-pointA[1])>0:
            return 100
        else:
            return -100
    return (pointB[1]-pointA[1])/(pointB[0]-pointA[0])

def pathSimplifier(path):
    THRESHOLD = 3
    THRESHOLD_CLOSE = 5
    index = 0
    simplePath = []
    simplePath.append(path[index])
    index = index + 1
    finalIndex = index
    while finalIndex < len(path)-1:
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
                if lookingFurther < THRESHOLD:
                    lookingFurther = lookingFurther+1
                else:
                    break
            else:
                lookingFurther = 0
                finalIndex = index+1
                if finalIndex > len(path)-1-THRESHOLD_CLOSE:
                    finalIndex = len(path)-1
                    break
            index = index + 1
                #print(f'Inside index: {index}')
                #print(f'Inside finalIndex: {finalIndex}')
        simplePath.append(path[finalIndex])

    return simplePath

def followPath(ourThymio, path):
    for index in range(len(path)-1):
        global REACHED = 0
        print(f'path index: {index}')
        if index == 0:
            angleToTurn = angleDifference(ourThymio.angle, angleTwoPoints(path[index+1],ourThymio.getCenter()))
            print(f'angleToTurn: {np.rad2deg(angleToTurn)}')
            #turnAngle(angleToTurn, ourThymio)
        else:
            angleToTurn = angleCalculatorPath(path[index],path[index-1],path[index+1])
            print(f'angleToTurn: {np.rad2deg(angleToTurn)}')
            #turnAngle(angleToTurn, ourThymio)

        stopKalmanFlag = Event()
        kThread = kalman.kalmanThread(stopFlag)
        
        distance = distanceCalculator(path[index], path[index+1])
        goForward(distance, ourThymio, stopFlag)
        
        kThread.start()
        while :
            
            #check if collision
            #check if local nav ended
            #do kalman
            iniVector = ourThymio.readKalman()
            tracker = kalman.createTracker(iniVector)
            tracker.predict(u=ourThymio.getVel())
            print(tracker)
            z = ourThymio.readKalman()
            tracker.update(z)
            print(tracker)


            time.sleep(5)
            # this will stop the timer
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            time.sleep(5)
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            print('salut')
            stopFlag.set()


'''def getSpeedConstant(robot_front_absolute, robot_center_absolute, destination_center_absolute, path, index, th, cap):
    #first rotation to put the thymio at 45/90 degrees
    angleToTurn = angleCalculator(robot_front_absolute, robot_center_absolute, destination_center_absolute)
    print(angleToTurn)
    turnAngle(angleToTurn, th)
    distanceInPixels = distanceCalculator(robot_center_absolute[0],robot_center_absolute[1], destination_center_absolute[0], destination_center_absolute[1])
    distanceOld = distanceInPixels
    print(f'distance ini: {distanceInPixels}')
    #start timer
    timeBefore = time.perf_counter()
    th.set_var("motor.left.target", 100)
    th.set_var("motor.right.target", 100)
    while int(distanceInPixels) > 5:
        frame = cap.read()
        robot_center_absolute, _ = vision.find_thymio_center(frame)
        distanceOld = distanceInPixels
        distanceInPixels = distanceCalculator(robot_center_absolute[0],robot_center_absolute[1], destination_center_absolute[0], destination_center_absolute[1])
        if np.isnan(distanceInPixels):
            distanceInPixels = distanceOld
        print(f'distance: {distanceInPixels}')
    th.set_var("motor.left.target", 0)
    th.set_var("motor.right.target", 0)
    print(f'total time: {time.perf_counter()-timeBefore}')
    #Go towards next point in path'''

'''    def getNextTurn(path,startIndex):
        index = startIndex
        refAngle = getAbsoluteAngle(path[index],path[index+1])
        straight = True
        index = index + 1
        while index < len(path)-1:
            #print(f'index:{index}')
            newAngle = getAbsoluteAngle(path[index], path[index+1])
            #print(abs(refAngle-newAngle))
            if abs(refAngle-newAngle) > 0.2:
                straight = False
                return index
            index = index + 1
        return index'''
