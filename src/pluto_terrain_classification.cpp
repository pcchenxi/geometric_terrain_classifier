#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

////////////////////////////// terrain related ///////////////////////////
#include "pointshape_processor.h"
#include "cloud_image_mapper.h"
#include "local_scan_buffer.h"
#include "feature_fuser.h"

tf::TransformListener* tfListener = NULL;
bool cloud_ready = false;
string target_frame = "world_corrected";

ros::Publisher  pub_path_roughness, pub_color, pub_height, pub_roughness;
image_transport::Publisher pub_img_color, pub_img_grey;

Pointshape_Processor *ps_processor;
Cloud_Image_Mapper   *ci_mapper;
Local_Scan_Buffer    *local_buff_height, *local_buff_cost;
Feature_Fuser        *feature_fuser;

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

    // float *vision_label;
    pcl::PointCloud<pcl::PointXYZRGB> colored_cloud = ci_mapper->cloud_image_mapping(image_msg, info_msg, velodyne_cloud);

    sensor_msgs::ImagePtr msg_color = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ci_mapper->img_label_color_).toImageMsg();
    sensor_msgs::ImagePtr msg_grey  = cv_bridge::CvImage(std_msgs::Header(), "mono8", ci_mapper->img_label_grey_).toImageMsg();

    pub_img_color.publish(msg_color);
    pub_img_grey.publish(msg_grey);

    // /////////////////////// get local scans from buffer /////////////////////////////
    // cout << "adding height, frame id: " << colored_cloud.header.frame_id;
    // local_buff_height->add_scan(colored_cloud);
    // cout << " adding cost, frame id: " << ps_processor->velodyne_cost.header.frame_id;
    // local_buff_cost->add_scan(ps_processor->velodyne_cost);

    // pcl::PointCloud<pcl::PointXYZRGB> local_cloud_height = local_buff_height->get_local_scans();
    // pcl::PointCloud<pcl::PointXYZRGB> local_cloud_cost   = local_buff_cost->get_local_scans();


    // publish(pub_height, local_cloud_height);
    // publish(pub_path_roughness, ps_processor->frontp_roughness);
    // publish(pub_roughness, local_cloud_cost);


    // feature_fuser->fusing(vision_label, ps_processor->cloud_feature, colored_cloud.points.size());
    // colored_cloud = feature_fuser->color_cloud_by_cost(colored_cloud, ps_processor->cloud_feature);
    // cout << colored_cloud.header.frame_id << endl;
    // publish(pub_color, colored_cloud);


//   try
//   {
//     cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
//     cv::waitKey(30);
//   }
//   catch (cv_bridge::Exception& e)
//   {
//     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
//   }

    // delete [] vision_label;
}

void callback_velodyne(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    cout << "in velodyne call back" << endl;
    tf::StampedTransform to_target;
    try {
        tfListener->lookupTransform(target_frame, "base_link", ros::Time(0), to_target);
    }catch(std::exception & e) {
       ROS_WARN_STREAM("TF error in pluto_terrain_classification! " << e.what());
        return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> filtered_velodyne_map = ps_processor->process_velodyne(cloud_in, tfListener);

    /////////////////////// get local scans from buffer /////////////////////////////
    cout << "adding height, frame id: " << filtered_velodyne_map.header.frame_id;
    if (!local_buff_height->add_scan(filtered_velodyne_map))
        return;

    cout << " adding cost, frame id: " << ps_processor->velodyne_cost.header.frame_id;
    if(!local_buff_cost->add_scan(ps_processor->velodyne_cost))
        return;

    pcl::PointCloud<pcl::PointXYZRGB> local_cloud_height = local_buff_height->get_local_scans();
    pcl::PointCloud<pcl::PointXYZRGB> local_cloud_cost   = local_buff_cost->get_local_scans();


    publish(pub_height, local_cloud_height);
    publish(pub_path_roughness, ps_processor->frontp_roughness);
    publish(pub_roughness, local_cloud_cost);


    velodyne_cloud = local_cloud_height;
    cloud_ready = true; 


}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pluto_terrain_classification");

    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);


    float cell_size = 0.2;
    int localscan_buff_size = 20;
    // string target_frame;
    node.getParam("/cell_size", cell_size);
    node.getParam("/localscan_buff_size", localscan_buff_size);
    // node.getParam("/target_frame", target_frame);

    ps_processor        = new Pointshape_Processor(360*4, cell_size);
    ci_mapper           = new Cloud_Image_Mapper(tfListener);
    local_buff_height   = new Local_Scan_Buffer(localscan_buff_size, target_frame);
    local_buff_cost     = new Local_Scan_Buffer(localscan_buff_size, target_frame);
    feature_fuser       = new Feature_Fuser();

    // pub_out         = node.advertise<sensor_msgs::PointCloud2>("/ground_obstacle",1);
    pub_path_roughness  = node.advertise<sensor_msgs::PointCloud2>("/velodyne_path_roughness",1);
    pub_roughness       = node.advertise<sensor_msgs::PointCloud2>("/velodyne_roughness",1);
    pub_height          = node.advertise<sensor_msgs::PointCloud2>("/velodyne_height",1);
    // pub_color           = node.advertise<sensor_msgs::PointCloud2>("/velodyne_colored",1);

    ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_velodyne);
    //image_transport::Publisher pub = it.advertise("camera/image", 1);
    
    image_transport::ImageTransport it(node);
    // image_transport::Subscriber sub = it.subscribe("image_raw", 1, imageCallback);

    image_transport::CameraSubscriber sub_camera;
    sub_camera = it.subscribeCamera("image_raw", 1, imageCallback);
    pub_img_color  = it.advertise("geometry_color", 1);
    pub_img_grey   = it.advertise("geometry_grey", 1);
    ros::spin();

    return 0;
}
