#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/transforms.h>

#include <terrain_classifier/cloud_matrix_loador.h>

tf::TransformListener* tfListener = NULL;
pcl::PointCloud<pcl::PointXYZRGB> ground_cloud_;
Cloud_Matrix_Loador* cml;

ros::Publisher  pub_cloud_geo;

float robot_x_, robot_y_;

string center_frame = "ego_rot";
string world_frame = "map";
string map_frame = world_frame;
string output_frame = "base_link_oriented";
string process_frame = world_frame;

bool _ready_to_start = true;

void publish(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZ> cloud, int type = 2)
{
    sensor_msgs::PointCloud2 pointlcoud2;
    pcl::toROSMsg(cloud, pointlcoud2);

    if(type == 2)
    { 
        pub.publish(pointlcoud2);
    }
    else
    {
        sensor_msgs::PointCloud pointlcoud;
        sensor_msgs::convertPointCloud2ToPointCloud(pointlcoud2, pointlcoud);

        pointlcoud.header = pointlcoud2.header;
        pub.publish(pointlcoud);
    }

}

void publish(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZRGB> cloud, int type = 2)
{
    sensor_msgs::PointCloud2 pointlcoud2;
    pcl::toROSMsg(cloud, pointlcoud2);

    pub.publish(pointlcoud2);
}

pcl::PointCloud<pcl::PointXYZRGB> transform_cloud(pcl::PointCloud<pcl::PointXYZRGB> cloud_in, sensor_msgs::PointCloud2 message_in, string frame_target)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    pcl::PointCloud<pcl::PointXYZRGB> cloud_out;
    tf::StampedTransform to_target;
    try 
    {
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, message_in.header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        // return cloud_in;
    }

    pcl_ros::transformPointCloud (cloud_in,cloud_out,to_target);  
    cloud_out.header.frame_id = frame_target;
    return cloud_out;
}

pcl::PointCloud<pcl::PointXYZ> transform_cloud(pcl::PointCloud<pcl::PointXYZ> cloud_in, sensor_msgs::PointCloud2 message_in, string frame_target)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    pcl::PointCloud<pcl::PointXYZ> cloud_out;
    tf::StampedTransform to_target;
    try 
    {
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, message_in.header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        // return cloud_in;
    }

    pcl_ros::transformPointCloud (cloud_in,cloud_out,to_target);  
    cloud_out.header.frame_id = frame_target;
    return cloud_out;
}

sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2 cloud_in, string frame_target, ros::Time stamp)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    sensor_msgs::PointCloud2 cloud_out;
    tf::StampedTransform to_target;

    try 
    {
        // tf_listener_->waitForTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, ros::Duration(1.0));
        // tf_listener_->lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        cloud_in.height = 0;
        cloud_in.width = 0;

        return cloud_in;
    }

    Eigen::Matrix4f eigen_transform;
    pcl_ros::transformAsMatrix (to_target, eigen_transform);
    pcl_ros::transformPointCloud (eigen_transform, cloud_in, cloud_out);

    cloud_out.header.frame_id = frame_target;
    return cloud_out;
}

pcl::PointCloud<pcl::PointXYZ> remove_cloud_outside_imgfov(pcl::PointCloud<pcl::PointXYZ> cloud_camera)
{
    pcl::PointCloud<pcl::PointXYZ> cloud_filtered;

    for(size_t i = 0; i < cloud_camera.points.size(); i++)
    {
        pcl::PointXYZ point_camera = cloud_camera.points[i];
        if(point_camera.z < 1.0)
            continue;

        cv::Point2d uv;
        cv::Point3d pt_cv(point_camera.x, point_camera.y, point_camera.z);
        uv = project3D_to_image(pt_cv);
      
        if(uv.x >= 0 && uv.x < 1920 && uv.y >= 0 && uv.y < 1080)
        {
            cloud_filtered.push_back(point_camera);
        }
    }

    cloud_filtered.header.frame_id = cloud_camera.header.frame_id;
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ> cloud_filter(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr  input_cloud       (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_passthrough (new pcl::PointCloud<pcl::PointXYZ>);

    cout << "before filter  " << input_cloud->points.size() << endl;

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (input_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-10, 10);
    pass.filter (*cloud_passthrough);
    // cout << "after x filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (cloud_passthrough);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-10, 10);
    pass.filter (*cloud_passthrough);
    // cout << "after y filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (cloud_passthrough);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-3, 3);
    pass.filter (*cloud_passthrough);

    return *cloud_passthrough;
}

void callback_cloud(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    cout << "cloud in" << endl;
    if(cloud_in->height == 0 && cloud_in->width == 0)
        return;

    // convert message cloud to pcl cloud 
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud_in, pcl_cloud_camera;
    pcl::fromROSMsg(*cloud_in, pcl_cloud_in);

    cout << "cloud recieved: " << ros::Time::now() << endl;

    // transfer cloud to comera frame and filter out the point outside of the image fov
    pcl::PointCloud<pcl::PointXYZ> cloud_base = transform_cloud(pcl_cloud_in, *cloud_in, "base_link");    
    pcl::PointCloud<pcl::PointXYZ> cloud_base_filtered = cloud_filter(cloud_base);
    // pcl::PointCloud<pcl::PointXYZ> cloud_process = transform_cloud(pcl_cloud_in, *cloud_in, process_frame);    

    pcl::PointCloud<pcl::PointXYZ> cloud_camera = transform_cloud(cloud_base_filtered, *cloud_in, "kinect2_rgb_optical_frame");
    pcl::PointCloud<pcl::PointXYZ> cloud_camera_filtered = remove_cloud_outside_imgfov(cloud_camera);
    pcl::PointCloud<pcl::PointXYZ> cloud_process_filtered = transform_cloud(cloud_camera_filtered, *cloud_in, process_frame);
    cout << "point number: " << cloud_process_filtered.points.size() << endl;
    ros::Time begin = ros::Time::now();

    // save transform to world
    tf::StampedTransform to_target;
    try 
    {
        tfListener->lookupTransform(world_frame, center_frame, cloud_in->header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("TF exception:\n%s", ex.what());
        return;
    }

    // process cloud
    robot_x_ = to_target.getOrigin().x();
    robot_y_ = to_target.getOrigin().y();

    ground_cloud_ = cml->process_cloud(cloud_process_filtered, cloud_process_filtered, 30, 30, 6, 0.1, 0.1, robot_x_, robot_y_);
    ground_cloud_.header.frame_id = process_frame;

    cout << ros::Time::now() - begin << "  loaded cloud *********************" << endl;

    publish(pub_cloud_geo, ground_cloud_);

}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "geometric_classifier");

    cml = new Cloud_Matrix_Loador();


    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);

    ros::Subscriber sub_cloud     = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_cloud);
    pub_cloud_geo      = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered_geometric", 1);

    ros::spin();

    return 0;
}
