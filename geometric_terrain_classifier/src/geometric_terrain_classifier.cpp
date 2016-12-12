#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <centauro_costmap/CostMap.h>

#include <terrain_classifier/cloud_matrix_loador.h>


tf::TransformListener* tfListener = NULL;
Cloud_Matrix_Loador* cml;

ros::Publisher  pub_cloud, pub_costmap1, pub_costmap2, pub_costmap3;
bool initialized = false;

pcl::PointCloud<pcl::PointXYZRGB> ground_cloud1_, ground_cloud2_, ground_cloud3_;
centauro_costmap::CostMap cost_map1_, cost_map2_, cost_map3_;

float robot_x_, robot_y_;

string map_frame = "map";
string output_frame = "base_link_oriented";
string process_frame = "map";

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
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
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

void convert_to_costmap(Mat height, Mat h_diff, Mat slope, Mat roughness, Mat cost, float resoluation, centauro_costmap::CostMap &cost_map, float robot_x, float robot_y)
{
    cost_map.cells_x = h_diff.cols;
    cost_map.cells_y = h_diff.rows;
    cost_map.resolution = resoluation;

    cost_map.origin_x = robot_x - 0.5 * cost_map.cells_x * resoluation;
    cost_map.origin_y = robot_y - 0.5 * cost_map.cells_y * resoluation;

    cost_map.height.resize(cost_map.cells_x * cost_map.cells_y);
    cost_map.height_diff.resize(cost_map.cells_x * cost_map.cells_y);
    cost_map.slope.resize(cost_map.cells_x * cost_map.cells_y);
    cost_map.roughness.resize(cost_map.cells_x * cost_map.cells_y);
    cost_map.semantic_cost.resize(cost_map.cells_x * cost_map.cells_y);

    cout << "map size: " << cost_map.cells_x << " " << cost_map.cells_y << endl;

    for(int row = 0; row < h_diff.rows; row ++)
    {
        for(int col = 0; col < h_diff.cols; col ++)
        {
            float cost_v        = cost.ptr<float>(row)[col]; 
            float height_v      = height.ptr<float>(row)[col];
            float height_diff   = h_diff.ptr<float>(row)[col];
            float slope_v       = slope.ptr<float>(row)[col];
            float roughness_v   = roughness.ptr<float>(row)[col];

            int index = row * h_diff.rows + col;

            if(cost_v == -1)
            {
                cost_map.height[index]       = std::numeric_limits<float>::quiet_NaN();;
                cost_map.height_diff[index]  = std::numeric_limits<float>::quiet_NaN();;
                cost_map.slope[index]        = std::numeric_limits<float>::quiet_NaN();;
                cost_map.roughness[index]    = std::numeric_limits<float>::quiet_NaN();;
            }
            else
            {
                cost_map.height[index]       = height_v;
                cost_map.height_diff[index]  = height_diff;
                cost_map.slope[index]        = slope_v;
                cost_map.roughness[index]    = roughness_v;
            }
        }
    }
}


void callback_cloud(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    cout << "cloud recieved: " << ros::Time::now() << endl;
    sensor_msgs::PointCloud2 cloud_transformed = transform_cloud(*cloud_in, process_frame);
    if(cloud_transformed.height == 0 && cloud_transformed.width == 0)
        return;

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(cloud_transformed, pcl_cloud);

    ros::Time begin = ros::Time::now();

    // save transform to world
    tf::StampedTransform to_target;
    try 
    {
        tfListener->lookupTransform("map", "ego_rot", cloud_in->header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("TF exception:\n%s", ex.what());
        return;
    }

    // process cloud

    robot_x_ = to_target.getOrigin().x();
    robot_y_ = to_target.getOrigin().y();

    ground_cloud1_ = cml->process_cloud(pcl_cloud, 12, 12, 6, 0.05, 0.015, robot_x_, robot_y_);
    ground_cloud1_.header.frame_id = process_frame;
    convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 0.05, cost_map1_, robot_x_, robot_y_);



    // process cloud
    // centauro_costmap::CostMap cost_map1, cost_map2, cost_map3;
    // cost_map1.header.frame_id = output_frame;
    // cost_map2.header.frame_id = output_frame;
    // cost_map3.header.frame_id = output_frame;
    // cost_map1.header.stamp = cloud_in->header.stamp;
    // cost_map2.header.stamp = cloud_in->header.stamp;
    // cost_map3.header.stamp = cloud_in->header.stamp; 

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered1 = cml->process_cloud(pcl_cloud, 12.0, 12.0, 6.0, 0.1, 0.015, robot_x, robot_y);
    // cloud_filtered1.header.frame_id = process_frame;
    // convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 0.2, cost_map1, robot_x, robot_y);

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered2 = cml->process_cloud(pcl_cloud, 25, 25, 6, 0.15, 0.01);
    // cloud_filtered2.header.frame_id = process_frame;
    // convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 0.2, cost_map2, robot_x, robot_y);

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered3 = cml->process_cloud(pcl_cloud, 30, 30, 6, 1.0, 0.015);
    // cloud_filtered3.header.frame_id = process_frame;
    // convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 1.0, cost_map3, robot_x, robot_y);

    // cout << "robot position : " << robot_x << " " << robot_y << endl;
    // pcl::transformPointCloud (cloud_filtered1, cloud_filtered1, eigen_transform);
    // pcl::transformPointCloud (cloud_filtered2, cloud_filtered2, eigen_transform);
    // pcl::transformPointCloud (cloud_filtered3, cloud_filtered3, eigen_transform);

    cout << ros::Time::now() - begin << "  loaded cloud *********************" << endl;

    publish(pub_cloud, ground_cloud1_); // publishing colored points with defalt cost function

    // pub_costmap1.publish(cost_map1);
    // pub_costmap2.publish(cost_map2);
    // pub_costmap3.publish(cost_map3);
    
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "geometric_classifier");

    cml = new Cloud_Matrix_Loador();


    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);

    ros::Subscriber sub_cloud  = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_cloud);
    // ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/ndt_map", 1, callback_cloud);
    pub_cloud      = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);

    pub_costmap1 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map1", 1);
    pub_costmap2 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map2", 1);
    pub_costmap3 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map3", 1);

    ros::spin();

    return 0;
}
