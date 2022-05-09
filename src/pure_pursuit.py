#!/usr/bin/env python


import rospy

import numpy as np

import time

import utils

import tf


from geometry_msgs.msg import PoseArray, PoseStamped

from visualization_msgs.msg import Marker, MarkerArray

from ackermann_msgs.msg import AckermannDriveStamped

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class PurePursuit(object):

    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.

    """

    def __init__(self):

        # self.odom_topic       = rospy.get_param("~odom_topic")

        self.odom_topic = "/odom"

        self.lookahead        = 2

        self.speed            = 2

        self.wrap             = 1

        self.wheelbase_length = 0.325 # from param.yaml of racecar_simulator

        self.trajectory  = utils.LineTrajectory("/followed_trajectory")

        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)

        self.pose_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)

        self.viz_pub = rospy.Publisher("/circle_viz", MarkerArray, queue_size=1)

        self.goal_sub = rospy.Subscriber(

            "/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.error_pub = rospy.Publisher("/error", Float64, queue_size=10)
        self.zero_pub = rospy.Publisher("/zero", Float64, queue_size=10)


    def trajectory_callback(self, msg):

        ''' Clears the currently followed trajectory, and loads the new one from the message

        '''

        self.trajectory.clear()

        self.trajectory.fromPoseArray(msg)

        self.trajectory.publish_viz(duration=0.0)

        rospy.loginfo("Trajectory loaded.")


    def goal_cb(self, msg):

        self.goal = [msg.pose.position.x, msg.pose.position.y]



    def odom_callback(self, msg):


        if len(self.trajectory.points) != 0:


            # Convert pose to 3 element array (x, y, theta)

            orientation = msg.pose.pose.orientation

            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]

            pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, tf.transformations.euler_from_quaternion(quaternion)[2]])

            # Find intersection of lookahead and trajectory

            closest_point = find_closest_point(pose, self.trajectory.points)
            error = ((pose[0:1] - closest_point[0])**2 + (pose[1:2] - closest_point[1])**2)**(1/2.)
            self.error_pub.publish(error)
            self.zero_pub.publish(0.)


            intersections = solve_intersections(closest_point, self.trajectory.points, self.lookahead)

            intersect_globalframe, intersect_carframe = best_intersection(intersections, pose)


            # Visualizations

            self.make_viz(closest_point, intersections, intersect_globalframe)


            # Calculate wheel angle to reach intersection

            curvature_radius = self.lookahead**2 / (2*intersect_carframe[0])

            steering_angle = -np.arctan(self.wheelbase_length / curvature_radius)


            # Make and publish the Ackermann Drive msg

            self.publish_drive(steering_angle)

            if closest_point[0] == self.trajectory.points[-1][0] and closest_point[1] == self.trajectory.points[-1][1]:

                self.speed = 0

                self.trajectory.clear()

                self.publish_drive(0)

                self.speed = 2


    def make_viz(self, closest_point, intersections, intersect_globalframe):

        viz = MarkerArray()

        viz.markers.append(make_circle(0,closest_point, self.lookahead, (0.5, 0, 255, 0)))

        viz.markers.append(make_circle(1,intersect_globalframe, 0.3, (0.5, 255, 0, 0)))

        for i in range(len(intersections)):

            viz.markers.append(make_circle(2+i,intersections[i], 0.5, (0.5, 0, 0, 255)))

        self.viz_pub.publish(viz)


    def publish_drive(self, steering_angle):

        drive_msg = AckermannDriveStamped()

        drive_msg.header.stamp =  rospy.Time.now()

        drive_msg.header.frame_id = "base_link"

        drive_msg.drive.steering_angle = steering_angle

        drive_msg.drive.speed = self.speed

        self.drive_pub.publish(drive_msg)



# -------- HELPER FUNCTIONS ------------


def find_closest_point(pose, trajectory_points):

    # finds closest point on trajectory to car pose (https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725)

    min_distance_squared = 1*10**10

    closest_point = None


    for i in range(len(trajectory_points)-1):

        start = np.array(trajectory_points[i])

        end = np.array(trajectory_points[i+1])

        start_to_end = end-start

        length_squared = np.dot(start_to_end,start_to_end)

        start_to_pose = pose[0:2]-start

        distance_along_segment = max(0, min(1, np.dot(start_to_end,start_to_pose)/ length_squared))

        projection = start + distance_along_segment*start_to_end

        distance_squared = (pose[0] - projection[0])**2 + (pose[1] - projection[1])**2

        if distance_squared < min_distance_squared:

            min_distance_squared = distance_squared

            closest_point = projection


    return closest_point


def solve_intersections(closest_point, trajectory_points, radius):

    # look at each line segment, find the points of intersection with lookahead circle (https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428)

    output = []

    for i in range(len(trajectory_points)-1):

        P1 = np.array(trajectory_points[i])

        V = np.array(trajectory_points[i+1]) - np.array(trajectory_points[i])

        a = np.dot(V,V)

        b = 2 * np.dot(V,P1 - closest_point)

        c = np.dot(P1, P1) + np.dot(closest_point, closest_point) - 2 * np.dot(P1,closest_point) - radius**2

        disc = b**2 - 4 * a * c

        if disc >= 0:

            sqrt_disc = np.sqrt(disc)

            t1 = (-b + sqrt_disc) / (2 * a)

            t2 = (-b - sqrt_disc) / (2 * a)

            if 0 <= t1 <= 1:

                output.append(P1 + t1*V)

            if 0 <= t2 <= 1:

                output.append(P1 + t2*V)

        if len(output) == 2:

            break

    return output


def best_intersection(intersections, pose):

    # given a list of lookahead intersections, returns the one in 'forward' direction (in car frame)

    best_dot = -1

    best_intersect = None

    forward = np.array([np.cos(pose[2]), np.sin(pose[2])])

    for point in intersections:

        car_to_point = point - pose[0:2]

        dot = np.dot(car_to_point, forward) / np.dot(car_to_point, car_to_point)

        if dot > best_dot:

            best_intersect = point

            best_dot = dot


    # convert point to vehicle coordinates

    car_point_sub = best_intersect - pose[0:2]

    car_point = np.array((np.sin(pose[2]) * car_point_sub[0] - np.cos(pose[2]) * car_point_sub[1], np.cos(pose[2]) * car_point_sub[0] + np.sin(pose[2]) * car_point_sub[1]))


    # best_interesect is global frame, car_point is in car frame

    return best_intersect, car_point


def make_circle(id, point, radius, color):

    marker = Marker()

    marker.header.frame_id = "map"

    marker.header.stamp = rospy.Time.now()

    marker.type = Marker.CYLINDER

    marker.action = Marker.ADD

    marker.id = id

    marker.pose.position.x = point[0]

    marker.pose.position.y = point[1]

    marker.pose.position.z = 1

    marker.scale.x = 2*radius

    marker.scale.y = 2*radius

    marker.scale.z = 0.1

    marker.color.a = color[0]

    marker.color.r = color[1]

    marker.color.g = color[2]

    marker.color.b = color[3]

    return marker



if __name__=="__main__":

    rospy.init_node("pure_pursuit")

    pf = PurePursuit()

    rospy.spin()
