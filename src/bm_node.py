#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Vector3, Pose2D
from module_gd.msg import PointedLocation
from basicmodutil_pkg import commBM

class GestureDetectionBM:
    def __init__(self):
        # BM parameters
        self.__basic_mod_name = rospy.get_param('basic_mod_name', 'module_gd')
        self.__debug = rospy.get_param('debug', False)
        
        #comm-topics
        top_comm_pub  = "/function/output"
        top_event_pub = "/event/output"
        top_comm_sub  = "/master/output"

        # comm pub & sub
        self.__comm_pub = rospy.Publisher(top_comm_pub,String,queue_size=1)
        self.__event_pub = rospy.Publisher(top_event_pub,String,queue_size=1)
        self.__comm_sub = rospy.Subscriber(top_comm_sub,String,self.__comm_cb)

        # intra-BM comm
        self.__start_gd_pub = rospy.Publisher('/module_gd/start_detection',String,queue_size=1)
        self.__floorptdet_sub = rospy.Subscriber('/module_gd/floor_pointing_results',PointedLocation,self.__fp_results_cb)

        # State variables
        self.__gesture_being_detected = None
    
    def __comm_cb(self,data):
        #Parse the function-invocation message into a dictionary
        bm, func, msgs = commBM.readFunCall(data.data)

        #Check if this basic module is being requested
        if(bm == self.__basic_mod_name):
            if(func == 'detectFloorPointing'):
                self.__detectFloorPointing()
    
    def __detectFloorPointing(self):
        if self.__gesture_being_detected == None:
            # Start process of detecting floor-pointing process
            self.__gesture_being_detected = 'floor-pointing'
            msg = String()
            msg.data = self.__gesture_being_detected
            self.__start_gd_pub.publish(msg)
        else:
            # Return empty response
            success = Bool(False)
            pointed_locs = PointedLocation()
            out_params = [success, pointed_locs]
            names = ['success', 'pointed_locs']
            msg_str = commBM.writeMsgFromRos(out_params, names)
            out_msg = String(msg_str)
            self.__comm_pub.publish(out_msg)

            if self.__debug:
                print('DEBUG: cannot detect floor-pointing, the BM is processing '+self.__gesture_being_detected)
    
    def __fp_results_cb(self,msg):
        # Reset state variable
        self.__gesture_being_detected = None

        # Return empty response
        success = Bool()
        success.data = (msg.hand != '') # An empty hand-name means failure
        pointed_loc = Pose2D()
        pointed_loc.x = msg.location.x
        pointed_loc.y = msg.location.y
        pointing_hand = String()
        pointing_hand.data = msg.hand
        out_params = [success, pointed_loc, pointing_hand]
        names = ['success', 'pointed_loc','pointing_hand']
        msg_str = commBM.writeMsgFromRos(out_params, names)
        out_msg = String(msg_str)
        self.__comm_pub.publish(out_msg)

if __name__ == '__main__':
    rospy.init_node('module_gd')
    GestureDetectionBM()
    rospy.spin()