#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include "cloud_matrix_loador.h"


tf::TransformListener* tfListener = NULL;
Cloud_Matrix_Loador* cml;

ros::Publisher  pub_cloud;


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


void callback_cloud(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    sensor_msgs::PointCloud2 cloud_transformed = transform_cloud(*cloud_in, "base_link");
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(cloud_transformed, pcl_cloud);

    ros::Time begin = ros::Time::now();

    pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered = cml->load_cloud(pcl_cloud);
    cloud_filtered.header.frame_id = pcl_cloud.header.frame_id;
    publish(pub_cloud, cloud_filtered);
    cout << ros::Time::now() - begin << "  loaded cloud" << endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "geometric_classifier");

    cml = new Cloud_Matrix_Loador(10, 10, 6, 0.03, 0.02);


    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);

    // ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/surfel_map/pointcloud", 1, callback_cloud);
    ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/ndt_map", 1, callback_cloud);
    pub_cloud = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);

    ros::spin();

    return 0;
}
