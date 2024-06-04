#!/usr/bin/env python3
import os
from ultralytics import YOLO
import cv2
import mediapipe as mp
import rospy
import rospkg
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from module_gd.msg import Vector3Array, Vector3Mat, CheckedVector3Array, CheckedVector3Mat
import message_filters
from math import tan, pi, sqrt
import numpy as np
from copy import copy

class JointsExtractor:
    def __init__(self):
        # Skeleton and people models
        mdc = rospy.get_param('min_detection_confidence', 0.5)
        mtc = rospy.get_param('min_tracking_confidence', 0.5)
        self.people_pose_model = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=mdc,
            min_tracking_confidence=mtc)
        yolo_model = rospy.get_param('yolo_model', 'yolov8n')
        assert yolo_model in ['yolov8n','yolov8s','yolov8m','yolov8l','yolov8x']
        r = rospkg.RosPack()
        self.model = YOLO(os.path.join(r.get_path('module_gd'),'models',yolo_model+'.pt'))
        
        # Skeleton-joint detection parameters
        self.subimg_margin = rospy.get_param('subimg_margin', 15)
        self.nonwrist_joint_radius = rospy.get_param('nonwrist_joint_radius', 5)
        self.wrist_joint_radius = rospy.get_param('wrist_joint_radius', 5)
        self.joint_seg_thresh = rospy.get_param('joint_seg_thresh', 0.15)
        self.scale_factor = rospy.get_param('scale_factor', 0.5)

        # RGBD-sensor parameters
        self.h_fov = rospy.get_param('horizontal_fov', 60.0)
        self.v_fov = rospy.get_param('vertical_fov', 49.5)
        self.sensor_frame = rospy.get_param('depthcam_link', 'none')
        self.sensor = rospy.get_param('sensor', 'none')
        
        # Input topics
        color_topic = rospy.get_param('color_topic', '/color_img')
        depth_topic = rospy.get_param('depth_topic', '/depth_img')
        
        # Publishers and subscribers
        # Color & depth images
        color_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.rgbdCB)
        # People joints
        self.joints_pub = rospy.Publisher('/module_gd/people_joints', CheckedVector3Mat, queue_size=10)

        # Intra-BM comm
        self.start_gd_sub = rospy.Subscriber('/module_gd/start_detection',String,self.start_gd_cb)

        # State variables
        self.gesture_being_detected = None
    
    def isInBound(self,img,x,y):
        if x > 0 and y > 0 and x < img.shape[1] and y < img.shape[0]:
            return True
        else:
            return False

    def depthRegionAvg(self,img,x,y,is_wrist):
        # Get all depth values within the neighborhood
        depth_vals = []
        vert = [1,-1]
        radius = None
        if is_wrist:
            radius = self.wrist_joint_radius
        else:
            radius = self.nonwrist_joint_radius
        radius_2 = radius*radius
        for i in range(x-radius,x+radius+1):
            if self.isInBound(img,i,y):
                if np.isnan(img[y,i,0]).any():
                    continue
                depth_vals.append(float(img[y,i,0]))
            else:
                continue
            for j in vert:
                tmp_y = copy(y)
                while True:
                    tmp_y = tmp_y + j
                    if self.isInBound(img,i,tmp_y):
                        if np.isnan(img[tmp_y,i,0]).any():
                            continue
                        euc_2 = (x-i)*(x-i) + (y-tmp_y)*(y-tmp_y)
                        if euc_2 < radius_2:
                            depth_vals.append(float(img[tmp_y,i,0]))
                        else:
                            break
                    else:
                        break

        # Sort them from closer to farthest and remove zeros
        depth_vals.sort()
        np_depth_vals = np.array(depth_vals,dtype=np.float64)
        np_depth_vals = np.delete(np_depth_vals,np.where(np_depth_vals < 0.3))
        
        # Get the values from the front cluster
        front_cluster = []
        for i in range(np_depth_vals.shape[0]):
            front_cluster.append(float(np_depth_vals[i]))
            if i < np_depth_vals.shape[0]-1:
                if abs(np_depth_vals[i]-np_depth_vals[i+1]) > self.joint_seg_thresh:
                    break

        # Compute the average of the closest cluster
        return np.mean(front_cluster)

    def prettyYOLOResults(self,results):
        names = []
        probs = []
        boxes = []
        for i in range(len(results)):
            pred = results[i].boxes.boxes
            for j in range(pred.shape[0]):
                names.append(results[i].names[int(pred[j,5])])
                probs.append(pred[j,4])
                tmp = [0,0,0,0]
                for k in range(4):
                    tmp[k] = int(pred[j,k])
                boxes.append(tmp)
        
        return names, probs, boxes

    def rgbdCB(self,color_msg,depth_msg):
        # Only process images if a gesture is being detected
        if self.gesture_being_detected == None:
            return

        # Convert to openCV images
        c_img = None
        d_img = None
        try:
            c_img = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
            c_img = cv2.cvtColor(c_img,cv2.COLOR_BGR2RGB)
            c_img = cv2.cvtColor(c_img,cv2.COLOR_RGB2BGR)
        except:
            return
        try:
            d_img = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
        except:
            try:
                d_img = np.frombuffer(depth_msg.data, dtype=np.float16).reshape(depth_msg.height, depth_msg.width, -1)
            except:
                return

        # Get bounding boxes of each person
        half_w = int(c_img.shape[1] * self.scale_factor)
        half_h = int(c_img.shape[0] * self.scale_factor)
        small_img = cv2.resize(c_img,(half_w,half_h))
        yolo_results = self.model.predict(source=small_img,classes=0)
        n,p,b = self.prettyYOLOResults(yolo_results)
        # Get sub-images and sub-image-offset of each person
        sub_images = []
        offsets = []
        for i in range(len(b)):
            x0 = int(max(b[i][0]*(1.0/self.scale_factor) - self.subimg_margin,0))
            y0 = int(max(b[i][1]*(1.0/self.scale_factor) - self.subimg_margin,0))
            x1 = int(min(b[i][2]*(1.0/self.scale_factor) + self.subimg_margin,c_img.shape[1]))
            y1 = int(min(b[i][3]*(1.0/self.scale_factor) + self.subimg_margin,c_img.shape[0]))
            offsets.append([x0,y0])
            sub_images.append(c_img[y0:y1,x0:x1].copy())

        # Obtain the person skeleton in each sub-image
        skeleton_cnt = 0
        shoulders_z = []
        people_joints = CheckedVector3Mat()
        for j in range(len(sub_images)):
            tmp = cv2.cvtColor(sub_images[j], cv2.COLOR_BGR2RGB)
            results = self.people_pose_model.process(tmp)
            if results.pose_landmarks:
                # Joint-list: 
                #   l-hip,r-hip
                #   l-shoulder,r-shoulder,
                #   l-elbow,r-elbow,
                #   l-wrist,r-wrist
                joints_index = [23,24,11,12,13,14,15,16]
                joints = CheckedVector3Array()
                sh_z = []
                for i in joints_index:
                    # Joint x,y ratio (i.e. [0,1]) in sub-image
                    sx = results.pose_landmarks.landmark[i].x
                    sy = results.pose_landmarks.landmark[i].y
                    # Joint x,y pixel in original image
                    px_x = int(sx * sub_images[j].shape[1]) + offsets[j][0]
                    px_y = int(sy * sub_images[j].shape[0]) + offsets[j][1]
                    # Joint x,y ratio (i.e. [0,1]) in original image
                    x = float(px_x) / float(c_img.shape[1])
                    y = float(px_y) / float(c_img.shape[0])
                    
                    # Make sure the joint-coordinate has valid values
                    v = Vector3()
                    f = Bool()
                    out_of_range = False
                    invalid_depth = False
                    # Check the joint's coordinate is in the image
                    out_of_range = not self.isInBound(c_img,px_x,px_y)
                    depth_val = np.nan
                    if not out_of_range:
                        is_wrist = (i == 15 or i == 16)
                        depth_val = self.depthRegionAvg(d_img,px_x,px_y,is_wrist)
                    # Check the depth value is a number
                    if not out_of_range:
                        if np.isnan(depth_val):
                            invalid_depth = True
                    if out_of_range or invalid_depth:
                        joints.vectors.append(v)
                        f.data = False
                        joints.flags.append(f)
                        continue

                    # Keep shoulders z to find the closest person
                    if i == 11 or i == 12:
                        sh_z.append(depth_val)

                    # Compute the joint's x-y-z coordinate (meters)
                    xm, ym = self.pxToMeters(x,y,depth_val)

                    # Adjust x-y-z corrdinate to the sensor's frame
                    if self.sensor_frame == 'azure_kinect_head_depth_camera_link':
                        v.z = depth_val
                        v.x = xm
                        v.y = ym * (-1.0)
                    elif self.sensor_frame == 'orbbec_astra_head_cam_depth_frame':
                        v.x = depth_val
                        v.z = ym
                        v.y = xm * (-1.0)
                    
                    # Store vectors
                    joints.vectors.append(v)
                    f.data = True
                    joints.flags.append(f)

                # Compute the shoulders z-dist
                if len(sh_z) == 0:
                    shoulders_z.append((skeleton_cnt,999.9))
                else:
                    shoulders_z.append((skeleton_cnt,sum(sh_z)/float(len(sh_z))))
                skeleton_cnt = skeleton_cnt + 1

                # Append the j-th person's joint x-y-z vectors
                people_joints.vectors.append(joints)

        # Sort list of people skeletons based on how close they are to the sensor (w.r.t. depth-axis)
        shoulders_z.sort(key=lambda a: a[1])
        sorted_people_joints = CheckedVector3Mat()
        for i in range(len(shoulders_z)):
            sorted_people_joints.vectors.append(people_joints.vectors[shoulders_z[i][0]])

        # Publish if people were found
        if len(sorted_people_joints.vectors) > 0:
            # Check that the closest person to the robot has enough info to detect where its pointing
            if self.gesture_being_detected == 'floor-pointing':
                has_l_elbow = (sorted_people_joints.vectors[0].vectors[4].z >= 0.0)
                has_r_elbow = (sorted_people_joints.vectors[0].vectors[5].z >= 0.0)
                has_l_wrist = (sorted_people_joints.vectors[0].vectors[6].z >= 0.0)
                has_r_wrist = (sorted_people_joints.vectors[0].vectors[7].z >= 0.0)
                if (has_l_elbow and has_l_wrist) or (has_r_elbow and has_r_wrist):
                    print('>> FOUND POINTING PERSON')
                    self.gesture_being_detected = None
                    self.joints_pub.publish(sorted_people_joints)

    def start_gd_cb(self,msg):
        # Trigger the detection of a gesture
        if self.gesture_being_detected == None:
            # Detection of where the closest person to the robot is pointing to the floor
            if msg.data == 'floor-pointing':
                self.gesture_being_detected = msg.data

    def pxToMeters(self,x,y,z):
        xm = (x - 0.5) * tan((self.h_fov/2.0) * pi / 180.0) * z
        ym = (0.5 - y) * tan((self.v_fov/2.0) * pi / 180.0) * z
        if self.sensor == 'orbbec':
            xm = xm * 2.0
            ym = ym * 2.0

        return xm, ym

if __name__ == '__main__':
    rospy.init_node('joints_extraction_node')
    JointsExtractor()
    rospy.spin()
