#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from Queue import PriorityQueue
import tf
from scipy import signal, ndimage


class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.DILATION_SIZE = 13
        self.init_set = False
        self.goal = None
        self.start = None
        self.x_dim = None
        self.y_dim = None
        self.meters_to_px = None
        self.origin = None
        self.rotation_matrix = None
        self.cell_map_constructed_flag = False
        self.start_set = False
        
        # possible sizes for building 31: [3, 7, 9, 11, 21, and up] pixels
        # possible sizes for stata: [5, 10, and up] pixels
        self.cell_size = 5 # cell_size x cell_size box  
        self.cell_map = None
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                         self.pose_init_cb,
                                         queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    class Cell(object):
        def __init__(self, occupied, row, col, cell_size):
            self.occupied = occupied
            self.index = (row, col)
            self.center = (row * cell_size + cell_size / 2, col * cell_size + cell_size / 2)
            self.cell_size = cell_size

        def contains(self, pixel):  # check if pixel is inside cell
            if self.center[0] - self.cell_size / 2 <= pixel[0] <= self.center[0] + self.cell_size / 2 and self.center[1] - self.cell_size / 2 <= pixel[1] <= self.center[1] + self.cell_size / 2:
                return True
            return False
        

################ CALLBACKS #######################

    def map_cb(self, map_msg):
        # Convert the map to a numpy array of Cells
        # self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.array(map_msg.data)
        self.x_dim = int(map_msg.info.width)
        self.y_dim = int(map_msg.info.height)
        self.map = np.reshape(self.map,
                              (self.y_dim, self.x_dim))
        self.map[self.map!=0] = 1
        self.map = ndimage.grey_dilation(self.map, size=(self.DILATION_SIZE, self.DILATION_SIZE)).astype(
            self.map.dtype)  # dilate the map to have more obstacles
        # self.map = np.clip(self.map, 0, 1)
        self.origin = map_msg.info.origin.position
        self.meters_to_px = float(1.0/map_msg.info.resolution)
        self.origin = map_msg.info.origin.position
        self.origin_pose = map_msg.info.origin
        orientation = map_msg.info.origin.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        theta = tf.transformations.euler_from_quaternion(
            [qx, qy, qz, qw])[2]
        self.rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta))))

        self.map = self.map.T  # so that the indeces of the array match the pixel indices
        # add_hstack = np.ones((self.x_dim, self.x_dim - self.y_dim + 8))
        add_hstack = np.ones((self.x_dim, self.x_dim - self.y_dim))
        # add_vstack = np.ones((8, 1738))
        self.map = np.hstack((self.map, add_hstack))
        # self.map = np.vstack((self.map, add_vstack))
        kernel = np.ones((self.cell_size, self.cell_size))
        occupancy = self.stride_conv(self.map, kernel, self.cell_size)
        occupancy[occupancy != 0] = 1
        # occupancy = np.clip(occupancy, 0, 1).astype(bool)
        occupancy = occupancy.astype(bool)
        self.cell_map = np.empty(occupancy.shape, dtype=object)
        vCell = np.vectorize(self.Cell)
        x_indices = [x for x in range(0, occupancy.shape[0]) for y in range(0, occupancy.shape[1])]
        y_indices = [y for x in range(0, occupancy.shape[0]) for y in range(0, occupancy.shape[1])]
        x_indices = np.array(x_indices).reshape(occupancy.shape)
        y_indices = np.array(y_indices).reshape(occupancy.shape)
        self.cell_map[:, :] = vCell(occupancy, x_indices, y_indices, self.cell_size)
        self.cell_map_constructed_flag = True

    def pose_init_cb(self, msg):
        #rospy.loginfo("Recieved pose from viz, defined particles")
        rospy.loginfo("Setting start.")
        self.start = self.tuple_to_pose((msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.start_set = True

    def odom_cb(self, msg):
        rospy.loginfo("Setting start.")
        self.start = self.tuple_to_pose((msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.start_set = True

    def goal_cb(self, msg):
        while not self.start_set or not self.cell_map_constructed_flag:
            rospy.sleep(2)
        rospy.loginfo("Setting goal.")
        self.goal = self.tuple_to_pose((msg.pose.position.x, msg.pose.position.y))
        self.plan_path(self.start, self.goal)

################ HELPERS #######################

    def stride_conv(self, arr, arr2, s):
        return signal.convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]

    def pix_to_map(self, coordinate):
        #Converts a coordinate from pixels to the map frame
        x_m = coordinate[0]/self.meters_to_px - self.origin.x # these were plus for the other map...
        y_m = coordinate[1]/self.meters_to_px - self.origin.y
        v = np.array([[x_m],[y_m]])
        rot = self.rotation_matrix.T.dot(v)  # maybe its the transpose bc i think we are given theta as the rotation of the pixel frame w respect to real map
        return tuple(rot)

    def map_to_pixel(self, pose):
        pixel_x = (pose.position.x - self.origin.x) 
        pixel_y = (pose.position.y - self.origin.y)
        v = np.array([[pixel_x], [pixel_y]])
        rot = np.dot(self.rotation_matrix, v)
        rot = np.round(rot * self.meters_to_px)
        return tuple(rot)

    def tuple_to_pose(self, node):
        p = Pose()
        p.position.x = np.round(node[0],2)
        p.position.y = np.round(node[1],2)
        return p

################ PATH-PLANNING #######################

    def get_neighbors(self, cell):
        neighbors = []
        cell_x = cell.index[0]
        cell_y = cell.index[1]
        width = self.cell_map.shape[0]
        height = self.cell_map.shape[1]
        indices = [(i, j) for i in [cell_x - 1, cell_x, cell_x + 1] for j in [cell_y - 1, cell_y, cell_y + 1] if i != j and 0 <= i < width and 0 <= j < height]
        for index in indices:
            neighbor = self.cell_map[index]
            if not neighbor.occupied:
                neighbors.append(neighbor)
        return neighbors

    def get_start_and_goal(self, start_pixel, goal_pixel):
        start = None
        goal = None
        for index, j in np.ndenumerate(self.cell_map):
            cell = self.cell_map[index]
            if start is None and cell.contains(start_pixel):
                start = cell
            if goal is None and cell.contains(goal_pixel):
                goal = cell
        return start, goal

    def get_distance(self, cell1, cell2):
        cell1_m = self.pix_to_map(cell1.center)
        cell2_m = self.pix_to_map(cell2.center)
        return np.asscalar(((cell1_m[1] - cell2_m[1])**2 + (cell1_m[0] - cell2_m[0])**2)**(1./2))

    def plan_path(self, start_point, goal_point):
        start_pixel = self.map_to_pixel(start_point)
        goal_pixel = self.map_to_pixel(goal_point)
        start_cell, goal_cell = self.get_start_and_goal(start_pixel, goal_pixel)
        expanded = set()

        came_from = dict()
        cost_so_far = dict()
        came_from[start_cell] = None
        cost_so_far[start_cell] = 0

        q = PriorityQueue()
        q.put((0, start_cell))

        while not q.empty():
            current = q.get()
            expanded.add(current[1])
            if current[1] is goal_cell:
                rospy.loginfo("Goal found!")
                break
            else:
                # rospy.loginfo("Analyzing edges.")
                neighbors = self.get_neighbors(current[1])
                for neighbor in neighbors:
                    if not neighbor.occupied:
                        new_cost = cost_so_far[current[1]] + self.get_distance(current[1], neighbor)
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            priority = new_cost + self.get_distance(neighbor, goal_cell)
                            if neighbor not in expanded:
                                q.put((priority, neighbor))
                            came_from[neighbor] = current[1]

        #extract the trajectory
        rospy.loginfo("Extracting trajectory.")
        current = goal_cell
        path = []
        while current is not start_cell:
            path.append(current)
            current = came_from[current]
        path.append(start_cell)
        path.reverse()
        
        self.trajectory = LineTrajectory("/planned_trajectory")   # initialized here so we can path plan over and over w/o rerunning
        for cell in path:
            cell_pose = self.tuple_to_pose(self.pix_to_map(cell.center))
            cell_point = Point()
            cell_point.x = cell_pose.position.x
            cell_point.y = cell_pose.position.y
            self.trajectory.addPoint(cell_point)

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

        print(goal_cell)
        print(goal_cell.index)
        print(goal_cell.center)


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
