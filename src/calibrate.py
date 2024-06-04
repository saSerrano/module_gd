#!/usr/bin/env python3
import os
import cv2
import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from module_gd.msg import Vector3Array, Vector3Mat
import message_filters
from math import tan, pi, sqrt, atan
import numpy as np
from copy import copy

class DepthCalibrator:
    def __init__(self):
        # in meters
        self.depth_thresh = 0.005

        # Input topics
        depth_topic = '/depth_to_rgb/hw_registered/image_rect_raw'

        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depthCB)

        self.got_wd = False

        self.d_img = None

        self.ends = {}

        # Calibration surface's dimensions (meters)
        self.surface_horizontal = 0.10
        self.surface_vertical = 0.15
    
    def depthCB(self,depth_msg):
        try:
            d_img = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
        except:
            try:
                d_img = np.frombuffer(depth_msg.data, dtype=np.float16).reshape(depth_msg.height, depth_msg.width, -1)
            except:
                return

        # Save most recent depth image
        self.d_img = d_img.copy()

        # Transform to displayble image
        #cp = d_img.copy()
        #mx = np.amax(cp)
        #cp = (cp / mx) * 255
        #cp = cp.astype(np.ubyte)
        cp = cv2.cvtColor(d_img,cv2.COLOR_GRAY2BGR)
        
        # Try to print the surface end points
        if 'l' in self.ends and 'r' in self.ends and 'u' in self.ends and 'd' in self.ends:
            for i in self.ends:
                pt = (self.ends[i][0],self.ends[i][1])
                ft = cv2.FONT_HERSHEY_SIMPLEX
                cp = cv2.putText(cp,'z'+i+':'+str(self.ends[i][2]),pt,ft,2.0,(255,0,0),2)
                cp = cv2.circle(cp, pt, 3, (0,0,255), 2)

        # Show depth image
        cv2.imshow('Depth',cp)
        if not self.got_wd:
            cv2.setMouseCallback('Depth',self.mouse_cb)
            self.got_wd = True
        cv2.waitKey(5)

    def pxToMeters(self,x,y,z):
        h_fov = 90.0
        v_fov = 59.0
        """ xm = (x - 0.5) * tan((self.h_fov/2.0) * pi / 180.0) * z * 2.0
        ym = (0.5 - y) * tan((self.v_fov/2.0) * pi / 180.0) * z * 2.0 """
        xm = (x - 0.5) * tan((h_fov/2.0) * pi / 180.0) * z * 2.0
        ym = (0.5 - y) * tan((v_fov/2.0) * pi / 180.0) * z * 2.0
        return xm, ym

    def mouse_cb(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find the left, right, upper and lower ends of the calibrating surface
            l = None
            r = None
            up = None
            lw = None
            # Left-most end
            prev_i = x
            prev_d = float(self.d_img[y,x,0])
            tmp = list(range(0,x))
            tmp.sort(reverse=True)
            for i in tmp:
                if abs(float(self.d_img[y,i,0]) - prev_d) > self.depth_thresh:
                    l = prev_i
                    break
                else:
                    prev_d = float(self.d_img[y,i,0])
                    prev_i = i
            # Right-most end
            prev_i = x
            prev_d = float(self.d_img[y,x,0])
            for i in range(x+1,self.d_img.shape[1]):
                if abs(float(self.d_img[y,i,0]) - prev_d) > self.depth_thresh:
                    r = prev_i
                    break
                else:
                    prev_d = float(self.d_img[y,i,0])
                    prev_i = i
            # Upper end
            prev_i = y
            prev_d = float(self.d_img[y,x,0])
            tmp = list(range(0,y))
            tmp.sort(reverse=True)
            for i in tmp:
                if abs(float(self.d_img[i,x,0]) - prev_d) > self.depth_thresh:
                    up = prev_i
                    break
                else:
                    prev_d = float(self.d_img[i,x,0])
                    prev_i = i
            # Lower end
            prev_i = y
            prev_d = float(self.d_img[y,x,0])
            for i in range(y+1,self.d_img.shape[0]):
                if abs(float(self.d_img[i,x,0]) - prev_d) > self.depth_thresh:
                    lw = prev_i
                    break
                else:
                    prev_d = float(self.d_img[i,x,0])
                    prev_i = i

            # Save in a dictionary the x,y,z values of the surface's ends
            self.ends['l'] = (l,y,float(self.d_img[y,l,0]))
            self.ends['r'] = (r,y,float(self.d_img[y,r,0]))
            self.ends['u'] = (x,up,float(self.d_img[up,x,0]))
            self.ends['d'] = (x,lw,float(self.d_img[lw,x,0]))

            # Compute FoV angles
            z = float(self.d_img[y,x,0])
            dx = abs(float(l-r)) / float(self.d_img.shape[1])
            dy = abs(float(up-lw)) / float(self.d_img.shape[0])
            h_fov = 180.0 * 2.0 * (1.0 / pi) * atan(self.surface_horizontal / (dx * z))
            v_fov = 180.0 * 2.0 * (1.0 / pi) * atan(self.surface_vertical / (dy * z))
            print('----')
            print('Horizontal FoV (deg):',h_fov)
            print('Vertical FoV (deg)  :',v_fov)
            print('z-center:',self.d_img[y,x,0])


if __name__ == '__main__':
    rospy.init_node('calibration_node')
    DepthCalibrator()
    rospy.spin()
