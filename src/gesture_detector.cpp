#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Bool.h>
#include <module_gd/Vector3Array.h>
#include <module_gd/Vector3Mat.h>
#include <module_gd/CheckedVector3Array.h>
#include <module_gd/CheckedVector3Mat.h>
#include <module_gd/PointedLocation.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <boost/signals2/mutex.hpp>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>

#include <tf/tf.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

using namespace std;

class GestureDetector
{
    private:

        boost::signals2::mutex mutex_;
        ros::Subscriber pose_sub_;

        tf2_ros::Buffer tf2_buffer_;
        tf2_ros::TransformBroadcaster tf2_br_;
        tf2_ros::TransformListener tf2_ls_;

        geometry_msgs::Pose pose_data_;
        string depthcam_link_;
        double min_dist_pointed_loc_;

        vector<unsigned int> hips_idx;
        vector<unsigned int> shoulders_idx;
        vector<unsigned int> elbows_idx;
        vector<unsigned int> wrists_idx;

        // Intra-comm
        ros::Publisher pointing_pub_;
        ros::Subscriber joints_sub_;

        bool debug_;
    
    public:

        GestureDetector(ros::NodeHandle nh_):
        tf2_buffer_(),
        tf2_ls_(tf2_buffer_)// Instantiate the transfrom listener
        {
            // Gather parameters
            string pose_topic;
            nh_.param<string>("pose_topic",pose_topic,"/pose");
            nh_.param<string>("depthcam_link",depthcam_link_,"orbbec_astra_head_cam_depth_frame");
            nh_.param<double>("min_dist_pointed_loc",min_dist_pointed_loc_,0.4);
            nh_.param<bool>("debug",debug_,false);

            // Check for valid parameters
            if(depthcam_link_ != "orbbec_astra_head_cam_depth_frame" && depthcam_link_ != "azure_kinect_head_depth_camera_link")
            {
                ROS_ERROR("Parameter depthcam_link must be \"orbbec_astra_head_cam_depth_frame\" or \"azure_kinect_head_depth_camera_link\".");
                return;
            }

            // Create subs and pubs
	        pointing_pub_ = nh_.advertise<module_gd::PointedLocation>(string("/module_gd/floor_pointing_results"),10);
            joints_sub_ = nh_.subscribe(string("/module_gd/people_joints"),10,&GestureDetector::jointsCB,this);
            pose_sub_ = nh_.subscribe(pose_topic,10,&GestureDetector::poseCB,this);

            // Lists of indexes (first is left, second is right)
            hips_idx.push_back(0);
            hips_idx.push_back(1);
            shoulders_idx.push_back(2);
            shoulders_idx.push_back(3);
            elbows_idx.push_back(4);
            elbows_idx.push_back(5);
            wrists_idx.push_back(6);
            wrists_idx.push_back(7);
        }

        vector<module_gd::CheckedVector3Array> rel2Map(vector<module_gd::CheckedVector3Array> const &vecs, geometry_msgs::Pose const &robot_pose)
        {
            //Variables temporales
            tf::Vector3 trans;
            tf::Quaternion rot;

            // (map) -> (base_footprint) transform
            tf::Transform m2bf_t;
            trans.setX(robot_pose.position.x);
            trans.setY(robot_pose.position.y);
            trans.setZ(robot_pose.position.z);
            rot = tf::Quaternion(robot_pose.orientation.x,
                                robot_pose.orientation.y,
                                robot_pose.orientation.z,
                                robot_pose.orientation.w);
            m2bf_t.setOrigin(trans);
            m2bf_t.setRotation(rot);

            // (base_footprint) -> (base_link) transform
            tf::Transform bf2bl_t;
            trans.setX(0.0);
            trans.setY(0.0);
            trans.setZ(0.025);
            rot = tf::Quaternion(0.0,0.0,0.0,1.0);
            bf2bl_t.setOrigin(trans);
            bf2bl_t.setRotation(rot);

            // Query (base_link) -> (depth-sensor link) transform
            tf::Transform bl2ds_t;
            geometry_msgs::TransformStamped tf2_t;
            bool got_tf = false;
            for(unsigned int i = 0; i < 10; i++)
            {
                try
                {
                    tf2_t = tf2_buffer_.lookupTransform("base_link",depthcam_link_,ros::Time(0));

                    trans.setX(tf2_t.transform.translation.x);
                    trans.setY(tf2_t.transform.translation.y);
                    trans.setZ(tf2_t.transform.translation.z);
                    rot = tf::Quaternion(tf2_t.transform.rotation.x,
                                        tf2_t.transform.rotation.y,
                                        tf2_t.transform.rotation.z,
                                        tf2_t.transform.rotation.w);
                    bl2ds_t.setOrigin(trans);
                    bl2ds_t.setRotation(rot);

                    got_tf = true;
                }
                catch(tf2::TransformException &ex)
                {   
                    cout << ex.what() << endl;
                    ros::Duration(0.1).sleep();
                    continue;
                }

                if(got_tf) break;
            }
            if(!got_tf) return vector<module_gd::CheckedVector3Array>();

            // (map) -> (depth-sensor link) transform
            tf::Transform m2ds_t;
            m2ds_t = m2bf_t * bf2bl_t* bl2ds_t;

            // Tranform vectors from relative to the depthcam to the map's frame
            vector<module_gd::CheckedVector3Array> map_vecs;
            for(unsigned int j = 0; j < vecs.size(); j++) // Iterate over people
            {
                module_gd::CheckedVector3Array person_joints;
                for(unsigned int i = 0; i < vecs[j].vectors.size(); i++)// Iterate over a person's joints
                {
                    // Vector in the map's frame
                    tf::Vector3 tmp;
                    tmp.setX(vecs[j].vectors[i].x);
                    tmp.setY(vecs[j].vectors[i].y);
                    tmp.setZ(vecs[j].vectors[i].z);
                    tmp = m2ds_t * tmp;
                    geometry_msgs::Vector3 m_vec;
                    m_vec.x = tmp.getX();
                    m_vec.y = tmp.getY();
                    m_vec.z = tmp.getZ();
                    person_joints.vectors.push_back(m_vec);
                    person_joints.flags.push_back(vecs[j].flags[i]);
                }
                map_vecs.push_back(person_joints);
            }

            return map_vecs;
        }

        bool floorPointedLocation(
            module_gd::CheckedVector3Array const &m_joints,
            bool const &left_hand,
            geometry_msgs::Vector3 &loc)
        {
            // Check that all necessary joints have valid values
            //condition ? expression1 : expression2;
            unsigned int tmp = (left_hand) ? 0 : 1;
            if(!m_joints.flags[elbows_idx[tmp]].data || !m_joints.flags[wrists_idx[tmp]].data) return false;

            // Compute the x and y line equations
            geometry_msgs::Vector3 elbow = m_joints.vectors[elbows_idx[tmp]];
            geometry_msgs::Vector3 wrist = m_joints.vectors[wrists_idx[tmp]];

            // (x,z) line equation
            double m_x = (wrist.z - elbow.z) / (wrist.x - elbow.x);
            double b_x = wrist.z - (m_x * wrist.x);
            // (x,y) line equation
            double m_y = (wrist.z - elbow.z) / (wrist.y - elbow.y);
            double b_y = wrist.z - (m_y * wrist.y);

            // Pointed location on the floor
            loc.x = -(b_x / m_x);
            loc.y = -(b_y / m_y);
            loc.z = 0.0;

            return true;
        }

        bool personLocation(module_gd::CheckedVector3Array const &m_joints, geometry_msgs::Vector3 &person_loc)
        {
            // Try to use the left and right shoulders' coordinates to compute the person's x,y location in the map
            double x(0.0);
            double y(0.0);
            int count(0);
            for(unsigned int i = 2; i <= 3; i++)
            {
                if(m_joints.flags[i].data)
                {
                    x += m_joints.vectors[i].x;
                    y += m_joints.vectors[i].y;
                    count++;
                }
            }
            bool can_compute_loc = (count > 0);
            
            // Compute the person's location if any shoulder data was avaiilable
            if(can_compute_loc)
            {
                person_loc.x = x / static_cast<double>(count);
                person_loc.y = y / static_cast<double>(count);
                person_loc.z = 0.0;
            }

            return can_compute_loc;
        }

        void jointsCB(const module_gd::CheckedVector3Mat::ConstPtr& msg)
        {
            // Get the person's joints and the robot's latest pose
            vector<module_gd::CheckedVector3Array> joints;
            for(unsigned int i = 0; i < msg->vectors.size(); i++)
            {
                joints.push_back(msg->vectors[i]);
            }
            geometry_msgs::Pose robot_pose = pose_data_;

            // Publish joints w.r.t. depth-camera
            if(debug_)
            {
                // Publish pointed locations (as TF) of each person's hands
                geometry_msgs::TransformStamped ts;
                ts.header.stamp = ros::Time::now();
                ts.header.frame_id = depthcam_link_;
                tf2::Quaternion q;
                q.setRPY(0, 0, 0);
                ts.transform.rotation.x = q.x();
                ts.transform.rotation.y = q.y();
                ts.transform.rotation.z = q.z();
                ts.transform.rotation.w = q.w();

                vector<string> jn;
                jn.push_back("L-H");
                jn.push_back("R-H");
                jn.push_back("L-S");
                jn.push_back("R-S");
                jn.push_back("L-E");
                jn.push_back("R-E");
                jn.push_back("L-W");
                jn.push_back("R-W");
                for(unsigned int i = 0; i < joints[0].vectors.size(); i++)
                {
                    ts.child_frame_id = jn[i];
                    ts.transform.translation.x = joints[0].vectors[i].x;
                    ts.transform.translation.y = joints[0].vectors[i].y;
                    ts.transform.translation.z = joints[0].vectors[i].z;
                    tf2_br_.sendTransform(ts);
                }
            }

            // Transform joints so they are in the map's frame
            vector<module_gd::CheckedVector3Array> map_joints;
            map_joints = rel2Map(joints,robot_pose);

            // Compute the location pointed by both arms
            vector<geometry_msgs::Vector3> left_loc, right_loc;
            vector<bool> left_good, right_good;
            for(unsigned int i = 0; i < map_joints.size(); i++)
            {
                geometry_msgs::Vector3 l_loc,r_loc;
                left_good.push_back(floorPointedLocation(map_joints[i],true,l_loc));
                right_good.push_back(floorPointedLocation(map_joints[i],false,r_loc));
                left_loc.push_back(l_loc);
                right_loc.push_back(r_loc);
            }

            // Determine which hand is pointing (if any)
            bool success(true);
            vector<geometry_msgs::Vector3> pointed_loc_vec{left_loc[0],right_loc[0]};
            vector<bool> status_loc_vec{left_good[0],right_good[0]};
            vector<string> name_vec{string("left"),string("right")};
            geometry_msgs::Vector3 person_loc;
            if(personLocation(map_joints[0],person_loc))
            {
                int pointing_hand(-1);
                double person2point_dist(0.0);
                for(unsigned int i = 0; i < pointed_loc_vec.size(); i++)
                {
                    if(status_loc_vec[i])
                    {
                        double dist = sqrt(pow(person_loc.x - pointed_loc_vec[i].x,2) + pow(person_loc.y - pointed_loc_vec[i].y,2));
                        if(dist >= min_dist_pointed_loc_ && dist > person2point_dist)
                        {
                            person2point_dist = dist;
                            pointing_hand = static_cast<int>(i);
                        }
                    }
                }

                // Check if a hand was pointing further than the min-dist radius
                if(pointing_hand >= 0)
                {
                    // Publish pointed location by the closest person to the robot 
                    module_gd::PointedLocation pl;
                    pl.hand = name_vec[pointing_hand];
                    pl.location = pointed_loc_vec[pointing_hand];
                    pointing_pub_.publish(pl);

                    // Publish joints w.r.t. depth-camera
                    if(debug_)
                    {
                        // Publish pointed locations (as TF) of each person's hands
                        geometry_msgs::TransformStamped ts;
                        ts.header.stamp = ros::Time::now();
                        ts.header.frame_id = "map";
                        tf2::Quaternion q;
                        q.setRPY(0, 0, 0);
                        ts.transform.rotation.x = q.x();
                        ts.transform.rotation.y = q.y();
                        ts.transform.rotation.z = q.z();
                        ts.transform.rotation.w = q.w();

                        ts.child_frame_id = "loc";
                        ts.transform.translation.x = pointed_loc_vec[pointing_hand].x;
                        ts.transform.translation.y = pointed_loc_vec[pointing_hand].y;
                        ts.transform.translation.z = 0.0;
                        tf2_br_.sendTransform(ts);
                    }
                }
                else
                {
                    ROS_WARN("Gesture-Detector: person is not pointing (not far enough of its location)");
                    success = false;
                }
            }
            else
            {
                ROS_WARN("Gesture-Detector: no data available to compute the person's location");
                success = false;
            }

            // If a pointed location could not be computed, publish failure status
            if(!success)
            {
                // Publish pointed location by the closest person to the robot
                module_gd::PointedLocation pl;
                pl.hand = "";
                pointing_pub_.publish(pl);
            }
        }

        void poseCB(const geometry_msgs::PoseStamped::ConstPtr& msg)
        {
            try
            {
                pose_data_ = msg->pose;
            }
            catch(exception& e)
            {
                cout << e.what() << endl;
            }
        }
};

int main(int argc, char** argv)
{
    ros::init(argc,argv,"gesture_detector_node");
    ros::NodeHandle nh_;
    GestureDetector gd(nh_);
    ros::spin();

    return 0;
}
