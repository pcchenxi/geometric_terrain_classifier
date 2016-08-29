#ifndef _FEATURE_FUSER_H_  
#define _FEATURE_FUSER_H_  

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>

#include "functions/terrain_function_sturcture.h"

using namespace std;

class Feature_Fuser
{
  public:
      Feature_Fuser();
      float max_cost;
      void fusing(float *vision_label, Feature *cloud_feature, int size);
      float compute_cost(float vision_label, float roughness);
      pcl::PointCloud<pcl::PointXYZRGB> color_cloud_by_cost(pcl::PointCloud<pcl::PointXYZRGB> cloud, Feature *cloud_feature);
};


Feature_Fuser::Feature_Fuser()
{

}

pcl::PointCloud<pcl::PointXYZRGB> Feature_Fuser::color_cloud_by_cost(pcl::PointCloud<pcl::PointXYZRGB> cloud, Feature *cloud_feature)
{
    for(int i = 0; i < cloud.points.size(); i ++)
    {
        int r = (int)(cloud_feature[i].cost/max_cost * 255);
        cloud.points[i].r = r;
    }

}

float Feature_Fuser::compute_cost(float vision_label, float roughness)
{
        cout << "vision_label: " << vision_label << " roughness: " << roughness << endl;
    float cost = vision_label * roughness;
    
    return cost; 
}

void Feature_Fuser::fusing(float *vision_label, Feature *cloud_feature, int size)
{
    for(int i = 0; i < size; i ++)
    { 
        if(cloud_feature[i].roughness == 0)
            continue;

        cout << "in susing: " << i << "vision_label: " << vision_label << cloud_feature[i].roughness << endl;
        // float cost = compute_cost(vision_label[i], cloud_feature[i].roughness);
        // cloud_feature[i].cost = cost;

        cout << "next " << endl;
        // if(max_cost < cost)
        //     max_cost = cost;
    }
}

#endif