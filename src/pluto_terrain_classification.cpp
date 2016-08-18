#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

////////////////////////////// terrain related ///////////////////////////
#include "pointshape_processor.h"
#include "cloud_image_mapper.h"
#include "local_scan_buffer.h"

tf::TransformListener* tfListener = NULL;
bool cloud_ready = false;
string target_frame = "map";

ros::Publisher  pub_out;
image_transport::Publisher pub_img;

Pointshape_Processor *ps_processor;
Cloud_Image_Mapper   *ci_mapper;
Local_Scan_Buffer    *local_buff;

pcl::PointCloud<pcl::PointXYZRGB> velodyne_cloud;

void publish(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZRGB> cloud, int type = 2)
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

sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2 cloud_in, string frame_target)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    sensor_msgs::PointCloud2 cloud_out;
    tf::StampedTransform to_target;

    try 
    {
        // tf_listener_->waitForTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, ros::Duration(1.0));
        // tf_listener_->lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, ros::Time(0), to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        // return cloud_in;
    }

    Eigen::Matrix4f eigen_transform;
    pcl_ros::transformAsMatrix (to_target, eigen_transform);
    pcl_ros::transformPointCloud (eigen_transform, cloud_in, cloud_out);

    cloud_out.header.frame_id = frame_target;
    return cloud_out;
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg,
               const sensor_msgs::CameraInfoConstPtr& info_msg)
{
    if(!cloud_ready)
        return;

    float *vision_label;
    pcl::PointCloud<pcl::PointXYZRGB> mapped_cloud = ci_mapper->cloud_image_mapping(image_msg, info_msg, velodyne_cloud, vision_label);
    // cout << mapped_cloud.header.frame_id << endl;
    publish(pub_out, mapped_cloud);
//   try
//   {
//     cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
//     cv::waitKey(30);
//   }
//   catch (cv_bridge::Exception& e)
//   {
//     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
//   }

 
}

void callback_velodyne(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    tf::StampedTransform to_target;
    try {
        tfListener->lookupTransform(target_frame, "base_link", ros::Time(0), to_target);
    }catch(std::exception & e) {
       ROS_WARN_STREAM("TF error in pluto_terrain_classification! " << e.what());
        return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> filtered_velodyne_base = ps_processor->process_velodyne(cloud_in, tfListener);
    // velodyne_cloud = filtered_velodyne_base;
    // cloud_ready = true;
    // cout << filtered_velodyne_base.header.frame_id << endl;

    /////////////////////// transfer filtered cloud from base_link to map frame ////////////////////
    pcl::PointCloud<pcl::PointXYZRGB> filtered_velodyne_map;
    Eigen::Matrix4f eigen_transform_target;
    pcl_ros::transformAsMatrix (to_target, eigen_transform_target);
    pcl::transformPointCloud(filtered_velodyne_base, filtered_velodyne_map, eigen_transform_target);
    filtered_velodyne_map.header.frame_id = target_frame;
    // cout << filtered_velodyne_map.header.frame_id << endl;

    /////////////////////// get local scans from buffer /////////////////////////////
    local_buff->add_scan(filtered_velodyne_map);
    pcl::PointCloud<pcl::PointXYZRGB> local_cloud = local_buff->get_local_scans();
    // publish(pub_out, local_cloud);


    // velodyne_cloud = local_buff->convert_to_pcl(*cloud_in);
    velodyne_cloud = local_cloud;
    cloud_ready = true;


}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pluto_terrain_classification");

    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);


    float cell_size = 0.2;
    int localscan_buff_size = 30;
    // string target_frame;
    node.getParam("/cell_size", cell_size);
    node.getParam("/localscan_buff_size", localscan_buff_size);
    // node.getParam("/target_frame", target_frame);

    ps_processor = new Pointshape_Processor(360*4, cell_size);
    ci_mapper    = new Cloud_Image_Mapper(tfListener);
    local_buff   = new Local_Scan_Buffer(localscan_buff_size, target_frame);

    pub_out = node.advertise<sensor_msgs::PointCloud2>("/ground_obstacle",1);

    ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_velodyne);
    // ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/ndt_map", 1, callback_velodyne);
    image_transport::ImageTransport it(node);
    // image_transport::Subscriber sub = it.subscribe("image_raw", 1, imageCallback);

    image_transport::CameraSubscriber sub_camera;
    sub_camera = it.subscribeCamera("image_raw", 1, imageCallback);
    pub_img = it.advertise("image_out", 1);
    ros::spin();

    return 0;
}
