#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from Queue import PriorityQueue
import tf
from scipy import signal

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        #self.odom_topic = rospy.get_param("~odom_topic")
        self.goal = None
        self.map = None
        self.map_set = False
        self.start_position = None
        self.init_set = False
        self.x_dim = None
        self.y_dim = None
        self.graph = None
        self.graph_set = False
        self.meters_to_px = None
        self.origin = None
        self.rotation_matrix = None
        self.rotation_matrix_ccw = None
        self.grid_size = .5 #sets the grid squares to be x meter x x meter
        self.grid_pix = None #the amount of pixels that make up an edge of a grid square
        self.box_x = 11
        self.box_y = 11


        self.odom_topic = "/odom"
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

    class Edge(object):
        def __init__(self, source, target, weight = 1):
            self.source = source
            self.target = target
            self.weight = weight

        def __hash__(self):
            return hash("%s_%s_%f" % (self.source, self.target, self.weight))

        def __eq__(self, other):
            return self.source == other.source and self.target == other.target \
                and self.weight == other.weight

    class Graph(object):
        def __init__(self):
            self._nodes = set()
            self._edges = dict()
        def __contains__(self, node):
            return node in self._nodes
        def add_node(self, node):
            self._nodes.add(node)
            #rospy.loginfo(node)
        def add_edge(self, node1, node2, weight = 1.0, bidirectional = False):
            self.add_node(node1)
            self.add_node(node2)
            node1_edges = self._edges.get(node1, set())
            node1_edges.add(PathPlan.Edge(node1, node2, weight))
            self._edges[node1] = node1_edges
            if bidirectional:
                node2_edges = self._edges.get(node2, set())
                node2_edges.add(PathPlan.Edge(node2, node1, weight))
                self._edges[node2] = node2_edges
        def node_edges(self, node):
            if not node in self:
                rospy.loginfo("Node not in graph.")
            # return self._edges.get(node, set())
            return self._edges[node]

    def strideConv(self, arr, arr2, s):
        return signal.convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]

    def map_cb(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        print(self.map.size)
        self.map = np.clip(self.map, 0, 1)
        print(np.mean(self.map))
        self.x_dim = int(map_msg.info.width)
        self.y_dim = int(map_msg.info.height)
        self.map = np.reshape(self.map, (self.y_dim, self.x_dim)).T # so that the indeces of the array match the pixel indices
        add_stack = np.ones((self.x_dim, self.x_dim - self.y_dim))
        self.map = np.hstack((self.map, add_stack))
        kernel = np.ones((self.box_y, self.box_x))
        occupancy = self.strideConv(self.map, kernel, 11)
        print(occupancy)
        print(occupancy.shape)
        occupancy = np.clip(occupancy, 0, 1)
        print(occupancy)
        print(np.mean(occupancy))



        print(self.x_dim)
        print(self.y_dim)
        self.meters_to_px = float(1.0/map_msg.info.resolution)
        print(self.meters_to_px)
        self.origin = map_msg.info.origin.position
        print(self.origin)
        orientation = map_msg.info.origin.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        theta = tf.transformations.euler_from_quaternion(
            [qx, qy, qz, qw])[2]
        print(theta)
        self.rotation_matrix = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
        self.map_set = True
        print(self.map.shape)
        print(self.map)

    def pix_to_map(self,coordinate):
        #Converts a coordinate from pixels to the map frame
        x_m = coordinate[0]/self.meters_to_px + self.origin.x
        y_m = coordinate[1]/self.meters_to_px + self.origin.y
        v = np.array((x_m,y_m))
        rot = self.rotation_matrix.dot(v)
        return tuple(rot)

    def map_to_graph(self, map):
        rospy.loginfo("Making graph.")
        self.graph = self.Graph()
        #Convert box to nearest odd pixel size so each box has a center pixel
        self.grid_pix = int(np.ceil(self.grid_size*self.meters_to_px)) # ceil(.5meter*19.99px/meter) = ceil(9.95) = 10 px^2 cells
        if self.grid_pix % 2 == 0:
            self.grid_pix = self.grid_pix + 1 
        #number of grid squares in x and y directions
        grid_x = self.x_dim/self.grid_pix
        grid_y = self.y_dim/self.grid_pix
        #Dimensions of map in pixels
        xdim_new = self.grid_pix*grid_x
        ydim_new = self.grid_pix*grid_y
        xs = np.linspace(self.grid_pix/2, xdim_new-self.grid_pix/2 - 1, grid_x) # midpoints of cells
        ys = np.linspace(self.grid_pix/2, ydim_new-self.grid_pix/2 - 1, grid_y)

        #Make the graph. The value at each index is the tuple of the pixel value in the map
        graph_array = np.asarray([[(i,j) for j in ys] for i in xs], dtype='int,int')
        #Make a duplicate sized one that keeps track of if cells are free for checking neighbors.
        #If it is 1, it's empty. If -1, it has an obstacle.
        #FREE IS NOT CURRENTLY BEING USED
        free = np.zeros((int(xdim_new),int(ydim_new)))

        rospy.loginfo("Checking free and building graph.")
        for index, cell in np.ndenumerate(graph_array):
            #Check if it or any of its surrounding pixels are occupied
            if self.is_free(map, cell, free):
                neighbors = self.get_neighbors(map,cell, free)
                cell_m = self.pix_to_map(cell)
                node = self.tuple_to_pose(cell_m)

                if len(neighbors) == 0:
                    self.graph.add_node(node)
                else:
                    a = np.array((cell_m[0],cell_m[1]))
                    for neighbor in neighbors:
                        neighbor_m = self.pix_to_map(neighbor)
                        b = np.array((neighbor_m[0],neighbor_m[1]))
                        weight = np.linalg.norm(a-b)
                        neighbor_node = self.tuple_to_pose(neighbor_m)
                        self.graph.add_edge(node,neighbor_node, weight)
        self.graph_set = True
        rospy.loginfo("Graph finished.")
        # print(self.graph._nodes)
        # for node  in self.graph._nodes:
        # 	if node not in self.graph._edges:
        # 		print("node not in edges?")
        # 	elif len(self.graph._edges[node]) == 0:
        # 		print("node has no edges?")


    def get_neighbors(self, map, cell, free):
        #Gets adjacent grid squares and removes the non-free ones.
        x = cell[0]
        y = cell[1]
        cell = (x,y)
        neighbors = [(i,j) for i in [x-self.grid_pix,x,x+self.grid_pix] for j in [y-self.grid_pix,y,y+self.grid_pix]]
        grid_x = self.x_dim/self.grid_pix
        grid_y = self.y_dim/self.grid_pix
        xdim_new = self.grid_pix*grid_x
        ydim_new = self.grid_pix*grid_y
        #Get rid of non valid neighbors -- if they aren't in the map or if they're not free
        neighbors = [z for z in neighbors if z[0] in range(xdim_new) if z[1] in range(ydim_new) if self.is_free(map, z, free) if z != cell]
        return neighbors

    def is_free(self, map, cell, free, tolerence = 0):
        #Checks if a cell on the map contains an obstacle. Cell is a tuple of the center pixel.
        #Tolerance tells you the lowest value to be marked as not an obstacle. Default 0.
            #convert from np.void (????)
            # cell = (cell[0],cell[1])
            #
            # if free[cell] == -1:
            #     return False
            # elif free[cell] == 1:
            #     return True
            # else:
            lr = self.grid_pix/2
            xs = np.linspace(cell[0] - lr, cell[0] + lr)
            ys = np.linspace(cell[1] - lr, cell[1] + lr)
            pix_array = np.asarray([[(i,j) for j in ys] for i in xs], dtype='int,int')
            for index, pix in np.ndenumerate(pix_array):
                index = self.y_dim*pix[0] + pix[1]
                if map[index] > tolerence:
                    # free[cell] = -1
                    return False
            # free[cell] = 1
            return True


    def tuple_to_pose(self, node):
        p = Pose()
        p.position.x = np.round(node[0],2)
        p.position.y = np.round(node[1],2)
        return p

    def odom_cb(self, msg):
        #rospy.loginfo("Called")
        if self.init_set:
            #rospy.loginfo("Init is already set.")
            return
        elif self.goal == None:
            rospy.loginfo("Goal not set.")
            return
        else:
            rospy.loginfo("Initial pose heard.")
            self.start_position = self.tuple_to_pose((msg.pose.pose.position.x, msg.pose.pose.position.y))
            self.init_set= True
            if not self.map_set:
                rospy.loginfo("map isn't set???")
            self.plan_path(self.start_position, self.goal, self.map)

    def goal_cb(self, msg):
        rospy.loginfo("Setting goal.")
        self.goal = self.tuple_to_pose((msg.pose.position.x, msg.pose.position.y))
        print("goal", self.goal)
        self.init_set = False

    def heuristic(self, current):
        #using euclidean distance
        point = np.array((current.position.x, current.position.y))
        goal = np.array((self.goal.position.x, self.goal.position.y))
        return np.linalg.norm(point-goal)

    def place_start_goal(self, start, goal):
        rospy.loginfo("Adding start and goal.")
        start_array = np.array((start.position.x, start.position.y))
        goal_array = np.array((goal.position.x, goal.position.y))
        start_dist = None
        goal_dist = None
        closest_start_node = None
        closest_goal_node = None
        nodes = self.graph._nodes
        for node in nodes:
            b = np.array((node.position.x, node.position.y))
            s_dist = np.linalg.norm(start_array-b)
            g_dist = np.linalg.norm(goal_array-b)
            if start_dist == None:
                start_dist = s_dist
                closest_start_node = node
            elif s_dist < start_dist:
                start_dist = s_dist
                closest_start_node = node
            if goal_dist == None:
                goal_dist = g_dist
                closest_goal_node = node
            elif g_dist < goal_dist:
                goal_dist = g_dist
                closest_goal_node = node
        self.graph.add_edge(start, closest_start_node, start_dist)
        self.graph.add_edge(goal, closest_goal_node, goal_dist)

    def plan_path(self, start_point, end_point, map):
        if not self.graph_set:
            self.map_to_graph(map)
        self.place_start_goal(start_point, end_point)
        #self.trajectory is the trajectory object.
        #assuming start point is a pose (point has .x, .y)
        rospy.loginfo("Starting path planning.")
        came_from = dict()
        cost_so_far = dict()
        came_from[start_point] = None
        cost_so_far[start_point] = 0
        q = PriorityQueue()
        q.put(start_point, 0)
        expanded = set()

        while not q.empty():
            current = q.get()
            expanded.add(current)
            if current not in self.graph:
                rospy.loginfo("Current not in graph.")
            else:
                if current == end_point:
                    rospy.loginfo("Goal found!")
                    break
                else:
                    rospy.loginfo("Analyzing edges.")
                    edges = self.graph.node_edges(current)
                    print(edges)
                    print(len(edges))
                    for i in edges:
                        # new_cost = cost_so_far[current] + i.weight
                        if i.target not in expanded:
                        # if i.target not in cost_so_far or new_cost < cost_so_far[i.target]:
                            new_cost = cost_so_far[current] + i.weight
                            if i.target not in cost_so_far or new_cost < cost_so_far[i.target]:
                                cost_so_far[i.target] = new_cost
                            priority = new_cost + self.heuristic(i.target)
                            q.put(i.target, priority)
                            rospy.loginfo(q)
                            came_from[i.target] = current
                        else:
                        	print("already expanded")

        #extract the trajectory
        rospy.loginfo("Extracting trajectory.")
        current = end_point
        path = []
        while current != start_point:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        for i in path:
            self.trajectory.addPoint(i)

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
