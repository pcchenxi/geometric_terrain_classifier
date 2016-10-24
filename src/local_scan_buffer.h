#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>

using namespace std;

class Local_Scan_Buffer
{
  public:
      Local_Scan_Buffer(int buffer_size, string buff_frame);
      int buffer_size_;
      int current_index_;
      string buff_frame_;
      std::vector<pcl::PointCloud<pcl::PointXYZRGB> > local_buffer_; 

      bool add_scan(pcl::PointCloud<pcl::PointXYZRGB> new_scan);
      bool add_scan(sensor_msgs::PointCloud2 new_scan);
      pcl::PointCloud<pcl::PointXYZRGB> get_local_scans();
      pcl::PointCloud<pcl::PointXYZRGB> convert_to_pcl(sensor_msgs::PointCloud2 cloud);
};


Local_Scan_Buffer::Local_Scan_Buffer(int buffer_size, string buff_frame)
{
    current_index_      = 0;
    buffer_size_        = buffer_size;
    buff_frame_         = buff_frame;
    local_buffer_.resize(buffer_size_);
}

bool Local_Scan_Buffer::add_scan(pcl::PointCloud<pcl::PointXYZRGB> new_scan)
{
    if(new_scan.header.frame_id != buff_frame_)
    {
        cout << "frame does not match with privious cloud frame !!!!" << buff_frame_ << "  " << new_scan.header.frame_id << endl;
        return false;
    }
    current_index_      ++;  
    if(current_index_ == buffer_size_)
        current_index_  = 0;

    local_buffer_[current_index_] = new_scan;

    return true;
}

pcl::PointCloud<pcl::PointXYZRGB> Local_Scan_Buffer::convert_to_pcl(sensor_msgs::PointCloud2 cloud)
{
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud_temp;
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;

    pcl::fromROSMsg(cloud, pcl_cloud_temp);
    copyPointCloud(pcl_cloud_temp, pcl_cloud); 

    pcl_cloud.header.frame_id = buff_frame_;
    return pcl_cloud;
}

bool Local_Scan_Buffer::add_scan(sensor_msgs::PointCloud2 new_scan)
{    
    if(new_scan.header.frame_id != buff_frame_)
    {
        cout << "frame does not match with privious cloud frame !!!!" << endl;
        return false;
    }

    pcl::PointCloud<pcl::PointXYZRGB> pcl_scan = convert_to_pcl (new_scan);

    current_index_      ++;  
    if(current_index_ == buffer_size_)
        current_index_  = 0;

    local_buffer_[current_index_] = pcl_scan;
    return true;
}

pcl::PointCloud<pcl::PointXYZRGB> Local_Scan_Buffer::get_local_scans()
{
    pcl::PointCloud<pcl::PointXYZRGB> local_scnas;

    for(int i = 0; i< local_buffer_.size(); i++)
    {
        local_scnas     += local_buffer_[i];
    }

    local_scnas.header.frame_id  = buff_frame_;

    if(local_scnas.points.size() == 0)
        std::cout << "empty !" << std::endl;

    return local_scnas;
}