#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/transforms.h>

#include <centauro_costmap/CostMap.h>
#include <terrain_classifier/cloud_matrix_loador.h>


tf::TransformListener* tfListener = NULL;
Cloud_Matrix_Loador* cml;

ros::Publisher  pub_cloud, pub_costmap1, pub_costmap2, pub_costmap3;


pcl::PointCloud<pcl::PointXYZRGB> ground_cloud1_, ground_cloud2_, ground_cloud3_;
centauro_costmap::CostMap cost_map1_, cost_map2_, cost_map3_;

string output_frame = "map";
string process_frame = "base_link_oriented";

bool initialized = false;

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

void set_output_frame(string output_frame, ros::Time stamp)
{
    cost_map1_.header.frame_id = output_frame;
    cost_map2_.header.frame_id = output_frame;
    cost_map3_.header.frame_id = output_frame;
    cost_map1_.header.stamp = stamp;
    cost_map2_.header.stamp = stamp;
    cost_map3_.header.stamp = stamp; 

    ground_cloud1_.header.frame_id = output_frame;
    ground_cloud2_.header.frame_id = output_frame;
    ground_cloud3_.header.frame_id = output_frame;
}

void callback_cloud(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    cout << "cloud recieved: " << ros::Time::now() << endl;
    sensor_msgs::PointCloud2 cloud_transformed = transform_cloud(*cloud_in, process_frame);
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(cloud_transformed, pcl_cloud);

    ros::Time begin = ros::Time::now();

    // save transform to world
    tf::StampedTransform to_target;
    try 
    {
        tfListener->lookupTransform(output_frame, pcl_cloud.header.frame_id, cloud_in->header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        return;
    }

    // process cloud

    float robot_x = to_target.getOrigin().x();
    float robot_y = to_target.getOrigin().y();

    ground_cloud1_ = cml->process_cloud(pcl_cloud, 12, 12, 6, 0.2, 0.015);
    ground_cloud1_.header.frame_id = process_frame;
    convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 0.025, cost_map1_, robot_x, robot_y);

    // ground_cloud2_ = cml->process_cloud(pcl_cloud, 5, 5, 6, 0.01, 0.015);
    // ground_cloud2_.header.frame_id = process_frame;
    // convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 0.05, cost_map2_, robot_x, robot_y);


    // ground_cloud3_ = cml->process_cloud(pcl_cloud, 30, 30, 6, 1.0, 0.015);
    // ground_cloud3_.header.frame_id = process_frame;
    // convert_to_costmap(cml->output_height_, cml->output_height_diff_, cml->output_slope_, cml->output_roughness_, cml->output_cost_, 1.0, cost_map3_, robot_x, robot_y);

    cout << "robot position : " << robot_x << " " << robot_y << endl;

    // transform point back to map frame
    Eigen::Matrix4f eigen_transform;
    pcl_ros::transformAsMatrix (to_target, eigen_transform);
    pcl::transformPointCloud (ground_cloud1_, ground_cloud1_, eigen_transform);
    pcl::transformPointCloud (ground_cloud2_, ground_cloud2_, eigen_transform);
    pcl::transformPointCloud (ground_cloud3_, ground_cloud3_, eigen_transform);

    set_output_frame(output_frame, cloud_in->header.stamp);

    cout << ros::Time::now() - begin << "  loaded cloud *********************" << endl;

    // publish(pub_cloud, ground_cloud2_); // publishing colored points with defalt cost function

    // pub_costmap1.publish(cost_map1);
    // pub_costmap2.publish(cost_map2);
    // pub_costmap3.publish(cost_map3);

    initialized = true;
    
}

  cv::Point2d project3D_to_image(cv::Point3d& xyz, string frame_id )
  {
    double fx, fy, cx, cy; 
    
    //   fx = 529.9732789120519;
    //   fy = 526.9663404399863;
    //   cx = 477.4416333879422;
    //   cy = 261.8692914553029;

    fx = 1060.707250708333;
    fy = 1058.608326305465;
    cx = 956.354471815484;
    cy = 518.9784429882449;

    cv::Point2d uv_rect;
    uv_rect.x = (fx*xyz.x) / xyz.z + cx;
    uv_rect.y = (fy*xyz.y) / xyz.z + cy;

    // cout << "projected uv: "<< xyz.x << " " << xyz.y << endl;
    return uv_rect;
  }

Mat image_cloud_mapper(const sensor_msgs::ImageConstPtr& image_msg, pcl::PointCloud<pcl::PointXYZRGB> ground_cloud, float map_width, float map_broad, float map_resolution)
{
    // init output image and transform pointcloud to camera frame
    string camera_frame = "kinect2_rgb_optical_frame_tune";
    // string camera_frame = image_msg->header.frame_id;
    pcl::PointCloud<pcl::PointXYZRGB> ground_cloud_camera, ground_cloud_base;

    Mat img_seg;
    Mat image_test = img_seg.clone();
    int img_rows = std::ceil(map_width/map_resolution);
    int img_cols = std::ceil(map_broad/map_resolution);
    Mat map_label = Mat(img_rows, img_cols, CV_32FC1,  Scalar(-1));
    // Mat map_label = Mat(img_rows, img_cols, CV_8UC1,  Scalar(0));

    if(ground_cloud.points.size() == 0)
        return map_label;
    
    try {
        img_seg = cv_bridge::toCvShare(image_msg, "32FC1")->image;
        // imshow("img_seg", img_seg);
        // waitKey(50);

        tf::StampedTransform to_camera;
        Eigen::Matrix4f eigen_transform_tocamera;
        tfListener->waitForTransform(camera_frame, ground_cloud.header.frame_id, image_msg->header.stamp, ros::Duration(0.15));
        tfListener->lookupTransform(camera_frame, ground_cloud.header.frame_id, image_msg->header.stamp, to_camera);
        pcl_ros::transformAsMatrix (to_camera, eigen_transform_tocamera);
        pcl::transformPointCloud (ground_cloud, ground_cloud_camera, eigen_transform_tocamera);

        tf::StampedTransform to_base;
        Eigen::Matrix4f eigen_transform_tobase;
        tfListener->waitForTransform(process_frame, ground_cloud.header.frame_id, image_msg->header.stamp, ros::Duration(0.15));
        tfListener->lookupTransform(process_frame, ground_cloud.header.frame_id, image_msg->header.stamp, to_base);
        pcl_ros::transformAsMatrix (to_base, eigen_transform_tobase);
        pcl::transformPointCloud (ground_cloud, ground_cloud_base, eigen_transform_tobase);
    }
    catch (cv_bridge::Exception& ex){
        cout << ex.what() << endl;
        return map_label; 
    } 

    cout << "ready for point projection " << camera_frame << " " << ground_cloud.header.frame_id << endl;
    // project points to image
    for(int i = 0; i < ground_cloud_camera.points.size(); i++)
    {
        pcl::PointXYZRGB point_camera = ground_cloud_camera.points[i];
        pcl::PointXYZRGB point_base   = ground_cloud_base.points[i];

        // cout << "projected uv: "<< ground_cloud.points[i].x << " " << point_camera.y << endl;
        if(point_camera.z < 0)
            continue;

        cv::Point2d uv;
        cv::Point3d pt_cv(point_camera.x, point_camera.y, point_camera.z);
        uv = project3D_to_image(pt_cv, camera_frame);
      
        // cout << "camera: " << point_camera.x << " " << point_camera.y << " " << point_camera.z << endl;
        // cout << "base: " << point_base.x << " " << point_base.y << " " << point_base.z << endl;
        // cout << "projected uv: "<< uv.x << " " << uv.y << endl;

        // check is the projected point inside image range
        if(uv.x >= 0 && uv.x < img_seg.cols && uv.y >= 0 && uv.y < img_seg.rows)
        {
            // cout << "projected uv: "<< uv.x << " " << uv.y << endl;
            float label_cost = img_seg.at<float>(uv.y, uv.x);
            // Vec3b label_cost = img_seg.at<Vec3b>(uv.y, uv.x);

            // compute index on the map
            point_base.x     += map_width/2;
            point_base.y     += map_broad/2;
            int col          =  point_base.x / map_resolution;
            int row          =  point_base.y / map_resolution;
            col              =  img_cols - col;

            // cout << "image: " << uv.y << " " << uv.x << endl;
            // cout << "map: " << row << " " << col << endl;
            if(col >= 0 && col < map_label.cols && row >= 0 && row < map_label.rows)
            {
                // cout << "label value: " << label_cost << endl;
                cv::circle(map_label, Point(col, row), 3, Scalar(label_cost), -1);  
                // cv::circle(map_label, Point(col, row), 3, Scalar(label_cost.val[0]), -1);  
                // map_label.at<float>(col, row) = label_cost;
                // map_label.at<Vec3b>(row, col) = label_cost;
                // ground_cloud.points[i].r = label_cost;
                // ground_cloud.points[i].g = label_cost;
                // ground_cloud.points[i].b = label_cost;
            }    

        }
    } 

    // imshow("map_label", map_label);

    // waitKey(50);

    cout << "projection finished" << endl;
    // ground_cloud_camera.header.frame_id = camera_frame;

    return map_label; 
}


void imageCallback_seg(const sensor_msgs::ImageConstPtr& image_msg)
{
    if(!initialized)
        return; 

    cout << "in image call back" << endl;
    Mat label_map = image_cloud_mapper(image_msg, ground_cloud1_, 12, 12, 0.2);

    // cout << "map: " << cost_map1_.cells_x << " " << cost_map1_.cells_y << endl;
    // cout << "image:  " << label_map.cols << " " << label_map.rows << endl;
    for(int row = 0; row < label_map.rows; row ++)
    {
        for(int col = 0; col < label_map.cols; col ++)
        {
            float semantic_v = label_map.ptr<float>(row)[col];

             int index = row * label_map.rows + col;

            if(semantic_v == -1)
                cost_map1_.semantic_cost[index] = std::numeric_limits<float>::quiet_NaN();
            else
                cost_map1_.semantic_cost[index] = semantic_v;
        }
    }

    publish(pub_cloud, ground_cloud1_);
    pub_costmap1.publish(cost_map1_);
    // pub_costmap2.publish(cost_map2_);
    // pub_costmap3.publish(cost_map3_);

    // imshow("label_map", label_map);
    // waitKey(20);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fused_classifier");

    cml = new Cloud_Matrix_Loador();


    ros::NodeHandle node; 
    tfListener = new (tf::TransformListener);

    ros::Subscriber sub_cloud     = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_cloud);
    ros::Subscriber sub_image_seg = node.subscribe<sensor_msgs::Image>("/image_seg", 1, imageCallback_seg);

    // ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/ndt_map", 1, callback_cloud);
    pub_cloud      = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);

    pub_costmap1 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map1", 1);
    pub_costmap2 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map2", 1);
    pub_costmap3 = node.advertise<centauro_costmap::CostMap>("/terrain_classifier/map3", 1);

    ros::spin();

    return 0;
}
