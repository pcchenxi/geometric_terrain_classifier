#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

sensor_msgs::PointCloud2 velodyne_cloud;

class FrameDrawer
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_;
  image_transport::Publisher pub_;
  ros::Publisher pub_velodyne_color_;
  tf::TransformListener tf_listener_;
  image_geometry::PinholeCameraModel cam_model_;
  std::vector<std::string> frame_ids_;
  CvFont font_;
  
  //sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2ConstPtr cloud_in, string frame_target);

public:
  FrameDrawer(const std::vector<std::string>& frame_ids)
    : it_(nh_), frame_ids_(frame_ids)
  {
    std::string image_topic = nh_.resolveName("image");
    
    sub_ = it_.subscribeCamera(image_topic, 1, &FrameDrawer::imageCb, this);
    pub_ = it_.advertise("image_out", 1);
    pub_velodyne_color_ = nh_.advertise<sensor_msgs::PointCloud2>("/velodyne_points_color", 1, this);

    // cout << image_topic << endl;
    //cvInitFont(&font_, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5);
  }


  sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2 cloud_in, string frame_target)
  {
    ////////////////////////////////// transform ////////////////////////////////////////
    sensor_msgs::PointCloud2 cloud_out;
    tf::StampedTransform to_target;
    
    // cout << frame_target << "  image frame: " << cloud_in.header.frame_id << endl;

    try 
    {
      //ros::Time acquisition_time = info_msg->header.stamp;
      //  ros::Duration timeout(1.0 / 1);
        //         tf_listener_.waitForTransform(cam_model_.tfFrame(), frame_id,
        //                               acquisition_time, timeout);
        // tf_listener_.lookupTransform(cam_model_.tfFrame(), frame_id,
        //                              acquisition_time, transform);
      tf_listener_.waitForTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, ros::Duration(1.0));
      tf_listener_.lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
      // tf_listener_.lookupTransform(frame_target, cloud_in.header.frame_id, ros::Time(0), to_target);
    }
    catch (tf::TransformException& ex) 
    {
      ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
      // return cloud_in;
    }
    
    // cout << frame_target << "   " << cloud_in.header.frame_id << endl;
    Eigen::Matrix4f eigen_transform;
    pcl_ros::transformAsMatrix (to_target, eigen_transform);
    pcl_ros::transformPointCloud (eigen_transform, cloud_in, cloud_out);
    
    cloud_out.header.frame_id = frame_target;
    return cloud_out;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg,
               const sensor_msgs::CameraInfoConstPtr& info_msg)
  {   

    cout << "image in" << endl;
    cv::Mat image, image_display;
    cv_bridge::CvImagePtr input_bridge;
    try {
      input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      image = input_bridge->image;
      image_display = image.clone();
    }
    catch (cv_bridge::Exception& ex){
      ROS_ERROR("[draw_frames] Failed to convert image");
      return;
    }
    
    cv::Mat img_mindist(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    
   // read camera information
    cam_model_.fromCameraInfo(info_msg);

    sensor_msgs::PointCloud2 cloud_transformed = transform_cloud (velodyne_cloud, cam_model_.tfFrame());
    //sensor_msgs::PointCloud2 cloud_transformed = transform_cloud (velodyne_cloud, "kinect2_ir_optical_frame");
    //pcl::PointCloud<pcl::PointXYZ> pcl_cloud_temp;
    //pcl::fromROSMsg(cloud_transformed, pcl_cloud_temp);
    
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl::fromROSMsg(cloud_transformed, pcl_cloud);
    //copyPointCloud(pcl_cloud_temp, pcl_cloud); 
  
    //BOOST_FOREACH(const std::string& frame_id, frame_ids_) 
    //{
    // string frame_id = frame_ids_[0];
    // cout << frame_id << endl;
    
    //tf::Point pt = transform.getOrigin();
    
    for(int i = 0; i < pcl_cloud.points.size(); i++)
    {
      pcl::PointXYZRGB point = pcl_cloud.points[i];
      if(point.z < 0)
        continue;
      
      cv::Point3d pt_cv(point.x, point.y, point.z);
      
      cv::Point2d uv;
      uv = cam_model_.project3dToPixel(pt_cv);

      
      static const int RADIUS = 25;
      //cout << uv << endl;
      
      if(uv.x >= 0 && uv.x < image.cols && uv.y >= 0 && uv.y < image.rows)
      {
        
        uchar min_z = img_mindist.at<uchar>(uv.y, uv.x);
        if(point.z < min_z)
        {
          cv::Point p1, p2;
          p1.x = uv.x;
          p1.y = uv.y - 5;
          p2.x = uv.x;
          p2.y = uv.y + 5;        
          //cv::circle(image, uv, RADIUS, CV_RGB(255,0,0), -1);
          cv::line(img_mindist, p1, p2, point.z/0.02, 5);
          
        }
         // cv::circle(img_mindist, uv, RADIUS, point.z/0.02, -1);
        
        // cv::Vec3b intensity = image.at<cv::Vec3b>(uv.y, uv.x);
        // uchar r = intensity.val[2];
        // uchar g = intensity.val[1];
        // uchar b = intensity.val[0];
        
        // pcl_cloud.points[i].r = r;
        // pcl_cloud.points[i].g = g;
        // pcl_cloud.points[i].b = b;
        
        // cv::Point p1, p2;
        // p1.x = uv.x;
        // p1.y = uv.y - 11;
        // p2.x = uv.x;
        // p2.y = uv.y + 11;        
        // //cv::circle(image, uv, RADIUS, CV_RGB(255,0,0), -1);
        // cv::line(image, p1, p2, CV_RGB(255,0,0), 1);
       
      }
    }
    
    for(int i = 0; i < pcl_cloud.points.size(); i++)
    {
      pcl::PointXYZRGB point = pcl_cloud.points[i];
      if(point.z < 0)
        continue;
      
      cv::Point3d pt_cv(point.x, point.y, point.z);
      
      cv::Point2d uv;
      uv = cam_model_.project3dToPixel(pt_cv);

      
      static const int RADIUS = 5;
      //cout << uv << endl;
      
      if(uv.x >= 0 && uv.x < image.cols && uv.y >= 0 && uv.y < image.rows)
      {
        uchar point_r = pcl_cloud.points[i].r;
        uchar point_g = pcl_cloud.points[i].g;
        uchar point_b = pcl_cloud.points[i].b;

        uchar min_z = img_mindist.at<uchar>(uv.y, uv.x);
        if(abs(point.z/0.02 - min_z) < 10 )
	      // if(point.z/0.02 < min_z )
        {
          cv::Vec3b intensity = image_display.at<cv::Vec3b>(uv.y, uv.x);

          uchar r = intensity.val[2];
          uchar g = intensity.val[1];
          uchar b = intensity.val[0];
          
          pcl_cloud.points[i].r = r;
          pcl_cloud.points[i].g = g;
          pcl_cloud.points[i].b = b;
              
          //cv::line(image, p1, p2, CV_RGB(255,0,0), 1);
        }
        cv::circle(image, uv, RADIUS, CV_RGB(point_r,point_g,point_b), -1);    
      }
    }
    
    cv::imshow("min", img_mindist);
    cv::waitKey(5);
        
    sensor_msgs::PointCloud2 cloud_pub;
    pcl::toROSMsg(pcl_cloud, cloud_pub);

    pub_velodyne_color_.publish(cloud_pub);

    //  break;
    //}
    
    pub_.publish(input_bridge->toImageMsg());
  }
};

void callback_velodyne(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    ///////////// transform ////////////////////
    velodyne_cloud = *cloud_in;
    // cout << "  velodyne frame: " << cloud_in->header.frame_id << endl;
    
  //   sensor_msgs::PointCloud2 cloud_transformed = transform_cloud (cloud_in, "base_link");
  //   cloud_transformed.header.frame_id = "base_link";
    
  //   pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  //  // pcl::fromROSMsg(*cloud_in, pcl_cloud);
  //   pcl::fromROSMsg(cloud_transformed, pcl_cloud);

}

int main(int argc, char** argv){
    ros::init(argc, argv, "cloud_image_mapper");
    ros::NodeHandle node;
    
    std::vector<std::string> frame_ids(argv + 1, argv + argc);
    
    //ros::Subscriber sub_velodyne = node.subscribe<sensor_msgs::PointCloud2>("/kinect2/sd/points", 10, callback_velodyne);
    //ros::Subscriber sub_velodyne = node.subscribe<sensor_msgs::PointCloud2>("ndt_map", 10, callback_velodyne);
    
    ros::Subscriber sub_velodyne = node.subscribe<sensor_msgs::PointCloud2>("points_raw", 10, callback_velodyne);
    
    cout << frame_ids[0] << endl;
    cout << frame_ids[1] << endl;    
    
    FrameDrawer drawer(frame_ids);
    ros::spin();
  
  
  
    // ros::NodeHandle node;
    // ros::Rate rate(10.0);

    // ros::Subscriber sub_velodyne = node.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, callback_velodyne);
    // ros::Subscriber sub_kinect = node.subscribe<sensor_msgs::PointCloud2>("/kinect2/sd/points", 10, callback_kinect);

    // pub_velodyne   = node.advertise<sensor_msgs::PointCloud2>("/velodyne_points_color", 1);

    // // ros::Subscriber sub_odom = node.subscribe<geometry_msgs::PoseStamped>("/slam_out_pose", 1, callback_odom);
    // // ros::Subscriber sub_odom_icp = node.subscribe<nav_msgs::Odometry >("/icp_odom", 1, callback_odom_icp);


    // tfListener = new (tf::TransformListener);

    // while (node.ok())
    // {

    //     ros::spinOnce();
    //     rate.sleep();
    // }
    // return 0;
};
