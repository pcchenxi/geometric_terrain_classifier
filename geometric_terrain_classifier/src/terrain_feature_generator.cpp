#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sstream>
#include <centauro_costmap/CostMap.h>
#include <terrain_classifier/geometric_feature_estimator.h>

tf::TransformListener* tfListener = NULL;
Geometric_Feature_Generator* gfg;

ros::Publisher  pub_cloud;
image_transport::Publisher pub_geometric_features;

string map_frame = "map";
string output_frame = "base_link_oriented";
string process_frame = "map";

Mat _final_label_img;
bool _ready_to_start = false;

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

sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2 cloud_in, string frame_target)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    sensor_msgs::PointCloud2 cloud_out;
    tf::StampedTransform to_target;
    try 
    {
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
      
        if(uv.x >= 0 && uv.x < 960 && uv.y >= 0 && uv.y < 540)
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
    if (_ready_to_start == false)
        return;
    else 
        _ready_to_start = false;

    if(cloud_in->height == 0 && cloud_in->width == 0)
        return;

    // convert message cloud to pcl cloud 
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud_in, pcl_cloud_camera;
    pcl::fromROSMsg(*cloud_in, pcl_cloud_in);
    
    cout << "cloud recieved: " << ros::Time::now() << endl;

    // transfer cloud to comera frame and filter out the point outside of the image fov
    pcl::PointCloud<pcl::PointXYZ> cloud_base = transform_cloud(pcl_cloud_in, *cloud_in, "base_link");    
    pcl::PointCloud<pcl::PointXYZ> cloud_base_filtered = cloud_filter(cloud_base);

    pcl::PointCloud<pcl::PointXYZ> cloud_camera = transform_cloud(cloud_base_filtered, *cloud_in, "kinect2_rgb_optical_frame");
    pcl::PointCloud<pcl::PointXYZ> cloud_camera_filtered = remove_cloud_outside_imgfov(cloud_camera);
    pcl::PointCloud<pcl::PointXYZ> cloud_process_filtered = transform_cloud(cloud_camera_filtered, *cloud_in, process_frame);
    cout << "point number: " << cloud_process_filtered.points.size() << endl;
    ros::Time begin = ros::Time::now();
    // cout << "cloud in " << pcl_cloud_in.header.frame_id << " cloud camera: " << cloud_camera_filtered.header.frame_id << " cloud process " << cloud_process_filtered.header.frame_id << endl;
    
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
    float robot_x = to_target.getOrigin().x();
    float robot_y = to_target.getOrigin().y();

    //// extract geometric feature from teh process_filtered cloud
    pcl::PointCloud<pcl::PointXYZRGB> ground_cloud;
    ground_cloud = gfg->process_cloud(cloud_process_filtered, 30, 30, 5, 0.1, 0.1, robot_x, robot_y);
    ground_cloud.header.frame_id = process_frame;
    convert_to_costmap(gfg->output_height_, gfg->output_height_diff_, gfg->output_slope_, gfg->output_roughness_, gfg->output_cost_, 0.1, cost_map1_, robot_x, robot_y);
    
    Mat features = gfg->generate_features_image(cloud_process_filtered, cloud_camera_filtered, gfg->cost_map_);
    sensor_msgs::ImagePtr img_features    = cv_bridge::CvImage(std_msgs::Header(), "32FC4", features).toImageMsg();
    pub_geometric_features.publish(img_features);

    cout << "done" << endl;
    cout << ros::Time::now() - begin << "  loaded cloud *********************" << endl;

    publish(pub_cloud, ground_cloud); // publishing colored points with defalt cost function  
}

void image_callback(sensor_msgs::ImageConstPtr image_msg)
{
    _final_label_img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    _ready_to_start = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "terrain_feature_generator");
    ros::NodeHandle node; 
    ros::NodeHandle private_nh("~");
    
    gfg = new Geometric_Feature_Generator();

    tfListener = new (tf::TransformListener);

    ros::Subscriber sub_cloud  = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_cloud);
    ros::Subscriber sub_image  = node.subscribe<sensor_msgs::Image>("/image_segmentation", 20, image_callback);
    pub_cloud      = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);

    image_transport::ImageTransport it(node);
    pub_geometric_features         = it.advertise("/gemoetric_features", 1);


    ros::spin();

    return 0;
}
