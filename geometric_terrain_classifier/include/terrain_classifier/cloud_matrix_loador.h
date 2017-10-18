#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <cmath>

using namespace std;
using namespace cv;

#define MIN_REQUIRED_H  1
#define GROUND_ACCURACY 0.05
 
#define L_SLOPE_R  0.10
#define S_SLOPE_R  0.05

class Cloud_Matrix_Loador
{

    float   map_resolution_;
    float   map_h_resolution_;
    float   map_width_;
    float   map_broad_;
    float   map_height_;

    int     map_rows_;
    int     map_cols_;
    int     map_hs_;

    float   map_resolution_output_;
    int     map_rows_output_;
    int     map_cols_output_;

    Mat     cloud_mat_; // 3D point space
    Mat     slope_map_l_, slope_map_s_, roughness_map_, height_map_, height_diff_map_, cost_map_; // 2D feature space

    public:
    Mat     output_height_diff_, output_slope_, output_roughness_, output_height_, output_cost_;

    float   max_roughness_;

    const float* mat_ptr;
    const float* slope_l_ptr;
    const float* slope_s_ptr;

    pcl::PointCloud<pcl::PointXYZ> cloud_cropped_, cloud_downsampled_;
    pcl::PointCloud<pcl::PointXYZ> ground_points_;
    pcl::PointCloud<pcl::PointXYZ> ground_points_index_;
    // pcl::PointCloud<pcl::PointXYZ> cloud_mat_index_;

    //   Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resalution, float map_h_resolution);
    Cloud_Matrix_Loador();
    ~Cloud_Matrix_Loador();

    void init_params(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution);

    void load_cloud(pcl::PointCloud<pcl::PointXYZ> cloud, float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution);
    pcl::PointCloud<pcl::PointXYZRGB> process_cloud    (pcl::PointCloud<pcl::PointXYZ> cloud, pcl::PointCloud<pcl::PointXYZ> cloud_to_color,
                                                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution, float robot_x, float robot_y);
    pcl::PointCloud<pcl::PointXYZ>    cloud_filter     (pcl::PointCloud<pcl::PointXYZ> cloud);
    pcl::PointCloud<pcl::PointXYZRGB> reformCloud      (pcl::PointCloud<pcl::PointXYZ> cloud, Mat cost_map);
    pcl::PointCloud<pcl::Normal>::Ptr calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                        float searchRadius );
    pcl::PointCloud<pcl::PointXYZ>    cloud_downsample (pcl::PointCloud<pcl::PointXYZ> cloud);                                               

    //////////////////////////////////////////////////
    pcl::PointCloud<pcl::PointXYZ>    get_ground_points(pcl::PointCloud<pcl::PointXYZ> *cloud);

    bool is_potential_ground                      (const float* h_ptr, int h);
    float get_vertical_mean_height                (const float* h_ptr, int h);
    Mat   rescaleMat                              (Mat img);


    ///////////////////////////////////////////// freture
    Mat  fill_missingvalue                        (Mat img);
    Mat  get_features                             (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt);
    Mat  get_feature_meanh                        (Mat img, int size);
    Mat  get_feature_roughness                    (Mat slope_l, Mat slope_s, Mat h_diff, Mat valid_mask, int size);
    void get_feature_slope_bycloud                (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt, Mat valid_mask);
    void get_feature_slope_byimgae                (Mat img_height, Mat valid_mask);
    Mat  compute_cost                             (Mat h_diff, Mat slope, Mat roughness, Mat valid_mask);
    Mat  generate_features_image                  (pcl::PointCloud<pcl::PointXYZ> cloud_base, pcl::PointCloud<pcl::PointXYZ> cloud_camera);
    
};

cv::Point2d project3D_to_image(cv::Point3d& xyz )
{
    double fx, fy, cx, cy; 

    // fx = 529.9732789120519;
    // fy = 526.9663404399863;
    // cx = 477.4416333879422;
    // cy = 261.8692914553029;

    // ubo
    fx = 1090.0084356403304;
    fy = 1093.4799569173076;
    cx = 944.9003536394688;
    cy = 527.4736145938574;

    cv::Point2d uv_rect;
    uv_rect.x = (fx*xyz.x) / xyz.z + cx;
    uv_rect.y = (fy*xyz.y) / xyz.z + cy;

    // cout << "projected uv: "<< xyz.x << " " << xyz.y << endl;
    return uv_rect;
}

// Cloud_Matrix_Loador::Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
Cloud_Matrix_Loador::Cloud_Matrix_Loador()
{
}

Cloud_Matrix_Loador::~Cloud_Matrix_Loador()
{

}

void Cloud_Matrix_Loador::init_params(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
{
    map_width_             = map_width;
    map_broad_             = map_broad;
    map_height_            = map_height;
    map_resolution_        = map_resolution/3;
    map_resolution_output_ = map_resolution;
    map_h_resolution_      = map_h_resolution;

    map_rows_              = std::ceil(map_width_/map_resolution_);
    map_cols_              = std::ceil(map_broad_/map_resolution_);
    map_rows_output_       = std::ceil(map_width_/map_resolution_output_);
    map_cols_output_       = std::ceil(map_broad_/map_resolution_output_);
    map_hs_                = std::ceil(map_height_/map_h_resolution);

    int sz[]               = {map_rows_, map_cols_, map_hs_};
    height_map_            = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_l_           = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_s_           = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    
    cloud_mat_             = Mat(3, sz, CV_32FC1,  Scalar(0.0));

    max_roughness_         = 0;

}


Mat Cloud_Matrix_Loador::generate_features_image(pcl::PointCloud<pcl::PointXYZ> cloud_base, pcl::PointCloud<pcl::PointXYZ> cloud_camera)
{
    cout << "in generate feature image function " << endl;
    Mat features_cam  = Mat(1080, 1920, CV_32FC4,  Scalar(0));
    for(size_t i = 0; i < cloud_base.points.size(); i++)
    {
        pcl::PointXYZ point_cam = cloud_camera.points[i];
        if(point_cam.z <= 1.0 )
            continue;

        // project points from camera fram to the image plane
        cv::Point3d pt_cv(point_cam.x, point_cam.y, point_cam.z);
        cv::Point2d uv = project3D_to_image(pt_cv);

        // cout << "after projection " << uv.x << " " << uv.y << " " << point_cam.x << " " << point_cam.y << " "  << point_cam.z << endl;
        // compute the position index from base frame
        pcl::PointXYZ point = cloud_base.points[i];
        point.x     += map_width_/2;
        point.y     += map_broad_/2;
        int col     = point.x / map_resolution_;
        int row     = point.y / map_resolution_;
        col         = map_cols_ - col;

        // int hit     = point.z / map_h_resolution_;
        if(row >= map_rows_ || col >= map_cols_ || row < 0 || col < 0)
            continue;

        float cost_obs = 3;
        float cost_rough = 2;
        float cost_flat = 1;

        float h = height_map_.ptr<float>(row)[col];
        float hd = height_diff_map_.ptr<float>(row)[col];
        float s = slope_map_l_.ptr<float>(row)[col];
        float r_1 = slope_map_s_.ptr<float>(row)[col];
        float r_2 = roughness_map_.ptr<float>(row)[col];

        Vec4f feature_value; 
        feature_value = features_cam.at<Vec4f>(uv.y, uv.x);

        if(feature_value[3] == 0 || feature_value[3] > point_cam.z)
        {
            feature_value[0] = hd;
            feature_value[1] = s;
            feature_value[2] = r_2;
            feature_value[3] = point_cam.z;
            features_cam.at<Vec4f>(uv.y, uv.x) = feature_value;
            // cout << features_cam.at<Vec4f>(uv.y, uv.x) << endl;
        }
        // cout << "save to mat"  << endl;
    }
    cout << "out generate feature image function " << endl;
    return features_cam;
}

pcl::PointCloud<pcl::PointXYZ> Cloud_Matrix_Loador::cloud_filter(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr  input_cloud       (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_passthrough (new pcl::PointCloud<pcl::PointXYZ>);

    cout << "before filter  " << input_cloud->points.size() << endl;

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (input_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-map_height_/2, map_height_/2);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_passthrough);
    // cout << "after z filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (cloud_passthrough);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-map_width_/2, map_width_/2);
    pass.filter (*cloud_passthrough);
    // cout << "after x filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (cloud_passthrough);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-map_broad_/2, map_broad_/2);
    pass.filter (*cloud_passthrough);
    cout << "after y filter  " << cloud_passthrough->points.size() << endl;

    // pcl::VoxelGrid<pcl::PointXYZ> sor;
    // sor.setInputCloud (cloud_passthrough);
    // sor.setLeafSize (0.01, 0.01, 0.01);
    // sor.filter (*cloud_passthrough);
    // cout << "after voxel filter  " << cloud_passthrough->points.size() << endl;

    return *cloud_passthrough;
}

pcl::PointCloud<pcl::PointXYZ> Cloud_Matrix_Loador::cloud_downsample(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr  input_cloud       (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_passthrough (new pcl::PointCloud<pcl::PointXYZ>);

    // pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    // outrem.setInputCloud(input_cloud);
    // outrem.setRadiusSearch(S_SLOPE_R);
    // outrem.setMinNeighborsInRadius (3);
    // outrem.filter (*cloud_passthrough);
    // cout << "after RadiusOutlierRemoval filter  " << cloud_passthrough->points.size() << endl;

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (input_cloud);
    sor.setLeafSize (map_resolution_, map_resolution_, map_h_resolution_);
    // sor.setLeafSize (map_resolution_ * 2, map_resolution_ * 2, map_h_resolution_ * 2);
    sor.filter (*cloud_passthrough);
    cout << "after voxel filter  " << cloud_passthrough->points.size() << endl;

    return *cloud_passthrough;
}

pcl::PointCloud<pcl::Normal>::Ptr Cloud_Matrix_Loador::calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                         pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                         float searchRadius )
{
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (input_point);
    ne.setSearchSurface(search_point);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    ne.setRadiusSearch (searchRadius);
    ne.setViewPoint (0, 0, 1.5);
    ne.compute (*cloud_normals);

    return cloud_normals;
}

pcl::PointCloud<pcl::PointXYZRGB> Cloud_Matrix_Loador::reformCloud(pcl::PointCloud<pcl::PointXYZ> cloud, Mat cost_map)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud_color;
    copyPointCloud(cloud, cloud_color);

    for(size_t i = 0; i < cloud_color.points.size(); i++)
    {
        // int row = ground_points_index_.points[i].y;
        // int col = ground_points_index_.points[i].x;

        pcl::PointXYZ point = cloud.points[i];
        point.x     += map_width_/2;
        point.y     += map_broad_/2;
        int col     = point.x / map_resolution_;
        int row     = point.y / map_resolution_;
        col         = map_cols_ - col;

        // int hit     = point.z / map_h_resolution_;
        // if(hit >= map_hs_ || row >= map_rows_ || col >= map_cols_ || hit < 0 || row < 0 || col < 0)
        //     continue;

        float cost = cost_map.ptr<float>(row)[col];

        // float max = 1;
        // if(cost > max)
        //     cost = max;
        // cloud_color.points[i].r = 255/max * cost; 

        float cost_obs = 2;
        float cost_rough = 1;
        float cost_flat = 0;
  
        if(cost == cost_obs)
        {
            cloud_color.points[i].r = 200;
        }    
        else if(cost == cost_rough)
        {
            cloud_color.points[i].r = 100.0;
            cloud_color.points[i].g = 255.0;
            cloud_color.points[i].b = 0.0;
        }   
        else if(cost == cost_flat)
        {
            cloud_color.points[i].r = 0.0;
            cloud_color.points[i].g = 0.0;
            cloud_color.points[i].b = 255.0;
        }  
        else if(cost == -1)
        {
            cloud_color.points[i].r = 0.0;
            cloud_color.points[i].g = 255.0;
            cloud_color.points[i].b = 255.0;
        }
        // else  // for testing cost values
        // {
        //     float theshold_1 = 0.01;
        //     float theshold_2 = 0.2;

        //     if(cost < theshold_1)
        //     {
        //         cloud_color.points[i].r = 0.0;
        //         cloud_color.points[i].g = 255.0/theshold_1 * cost;
        //         cloud_color.points[i].b = 0.0; 
        //     }
        //     else if(cost < theshold_2)
        //     {
        //         cloud_color.points[i].r = 255.0/theshold_2 * cost;
        //         cloud_color.points[i].g = 255.0;
        //         cloud_color.points[i].b = 0.0; 
        //     }
        //     else
        //     {
        //         cloud_color.points[i].r = 255.0;
        //         cloud_color.points[i].g = 255.0;
        //         cloud_color.points[i].b = 255.0 * cost; 
        //     }
        // }
    }

    return cloud_color;
}

Mat Cloud_Matrix_Loador::compute_cost(Mat h_diff, Mat slope, Mat roughness, Mat valid_mask)
{
    Mat cost_map = Mat(h_diff.rows, h_diff.cols, CV_32FC1,  Scalar(0));
    for(int row = 0; row < h_diff.rows; row ++)
    {
        for(int col = 0; col < h_diff.cols; col ++)
        {
            float is_valid      = valid_mask.ptr<float>(row)[col];
            if(is_valid == 0)
            {
                cost_map.at<float>(row, col) = -1;
                continue;
            }    

            float height_diff   = h_diff.ptr<float>(row)[col];
            float slope_v       = slope.ptr<float>(row)[col];
            float roughness_v   = roughness.ptr<float>(row)[col];

            // if(roughness_v != roughness_v)
            // {
            //     cout << "roughness NAN !  " << roughness_v << endl;
            //     cost_map.at<float>(row, col) = 255.0;
            //     continue;
            // }    
            // if(slope_v != slope_v)
            //     cout << "slope NAN !" << endl;

            float cost = slope_v * 0.0 + roughness_v * 1.0;

            if(cost > max_roughness_)
                max_roughness_ = cost;

        
            if(height_diff > 0.25 || cost > 0.1)
            {
                cost = 2.0;   // obstacle
            }    
            else if(cost > 0.001)
                cost = 1.0;  // rough
            else 
                cost = 0;  // flat

            cost_map.at<float>(row, col) = cost;

        }
    }

    // cost_map = cost_map & valid_mask;
    return cost_map;
}

void Cloud_Matrix_Loador::load_cloud(pcl::PointCloud<pcl::PointXYZ> cloud, float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)   // load point cloud to cloud_mat_
{
    ros::Time begin = ros::Time::now();

    init_params(map_width, map_broad, map_height, map_resolution, map_h_resolution);
    /////////////////////// filter cloud //////////////////////
    // cloud_cropped_      = cloud_filter(cloud);
    // cloud_downsampled_  = cloud_downsample(cloud_cropped_);
    // cloud_downsampled_  = cloud_cropped_;
    pcl::PointCloud<pcl::PointXYZ> cloud_cropped;

    ros::Time t1 = ros::Time::now();
    cout <<  t1 - begin << " ------------------ init space "  << endl;
    // cloud_mat_index_.points.resize(cloud_cropped_.points.size());

    for(size_t i = 0; i < cloud.points.size(); i=i+1)
    {
        pcl::PointXYZ point = cloud.points[i];
        if(point.z > 5.0 || (abs(point.x) < 1.0 && abs(point.y < 1.0)))
            continue;

        point.x     += map_width_/2;
        point.y     += map_broad_/2;
        point.z     += map_height_/2;

        // cout << "start" << endl;
        int col     = point.x / map_resolution_;
        int row     = point.y / map_resolution_;
        int hit     = point.z / map_h_resolution_;

        col         = map_cols_ - col;

        // cout << row << " " << col << " " << hit << " " << map_rows_ << " " << map_cols_ << " " << map_hs_ << endl;
        if(hit >= map_hs_ || row >= map_rows_ || col >= map_cols_ || hit < 0 || row < 0 || col < 0)
            continue;

        if(cloud_mat_.ptr<float>(row, col)[hit] == 0)
        {
            cloud_cropped.points.push_back(cloud.points[i]);
        }
        cloud_mat_.ptr<float>(row, col)[hit] += 1;

        // cloud_mat_index_.points[i].x = col;
        // cloud_mat_index_.points[i].y = row;
        // cloud_mat_index_.points[i].z = hit;
        // cout << i << " " << cloud_mat_.at<float>(row, col, hit) << endl;
    }

    cloud_cropped_ = cloud_cropped;
    cloud_downsampled_  = cloud_cropped;
    // Mat subImg = cloud_mat_(cv::Range(0, 100), cv::Range(0, 100));
    // cout << countNonZero(subImg) << endl;
    ros::Time t2 = ros::Time::now();
    cout <<  t2 - t1 << " ------------------ loaded points to Mat: "  << cloud_cropped.points.size() << endl;
}

pcl::PointCloud<pcl::PointXYZRGB> Cloud_Matrix_Loador::process_cloud(pcl::PointCloud<pcl::PointXYZ> cloud, pcl::PointCloud<pcl::PointXYZ> cloud_to_color, 
                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution, float robot_x, float robot_y)
{
	for(size_t i = 0; i < cloud.points.size(); i++)
	{
		cloud.points[i].x -= robot_x;
        cloud.points[i].y -= robot_y;
        
		cloud_to_color.points[i].x -= robot_x;
        cloud_to_color.points[i].y -= robot_y;
	}
	
    ros::Time begin = ros::Time::now();

    load_cloud(cloud, map_width, map_broad, map_height, map_resolution, map_h_resolution);

    ground_points_ = get_ground_points(&cloud_cropped_);

    ros::Time t3 = ros::Time::now();
    cout << t3 - begin << " ------------------ detected ground points: "  << ground_points_.size() << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prt           (new pcl::PointCloud<pcl::PointXYZ>(cloud_downsampled_));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt    (new pcl::PointCloud<pcl::PointXYZ>(ground_points_));
    cost_map_ = get_features(cloud_prt, cloud_ground_prt);

    ros::Time t4 = ros::Time::now();
    cout << t4 - t3 << " ------------------ computed features: "  << endl;

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_color = reformCloud(ground_points_, cost_map_); // only color ground points
    pcl::PointCloud<pcl::PointXYZRGB> cloud_color = reformCloud(cloud_to_color, cost_map_); // color all points
    
    ros::Time t5 = ros::Time::now();
    cout << t5 - t4 << " ------------------finished reformCloud: "  << cloud_color.points.size() << endl;

	for(size_t i = 0; i < cloud_color.points.size(); i++)
	{
		cloud_color.points[i].x += robot_x;
        cloud_color.points[i].y += robot_y;       
	}
	
    // pcl::PointCloud<pcl::PointXYZRGB> cloud_color;
    // return cloud_color;
    return cloud_color;
}



bool Cloud_Matrix_Loador::is_potential_ground(const float* h_ptr, int h)
{
    int min_required_step   = MIN_REQUIRED_H / map_h_resolution_;
    int empty_count         = 0;

    bool is_ground          = false;

    // cout << "in potentional ground: " << h << " ";
    for(int height = h+1; height < map_hs_; height ++)
    {
        float num = h_ptr[height];

        // cout << " " << height << ": " << num ;

        if(num == 0)
        {
            empty_count ++;
        } 
        else
            break;

        if(empty_count >= min_required_step)
        {
            is_ground = true;
            break;
        }
    }

    // cout << " " << empty_count << endl;
    return is_ground;
}

float Cloud_Matrix_Loador::get_vertical_mean_height(const float* h_ptr, int h)
{
    int sum = 0;
    int count = 0;
    // int max_num = 0;
    // int max_h = 0;
    int mean_step = GROUND_ACCURACY / map_h_resolution_;

    int start_index = h + mean_step;
    int end_index = h - mean_step;
    if(end_index  < 0)
        end_index = 0;
    if(start_index  >= map_hs_)
        start_index = map_hs_ -1;
        
    for(int height = start_index; height >= end_index; height --)
    {
        float num = h_ptr[height];
        if(num == 0)
            continue;
        
        // if(num > max_num)
        // {
        //     max_h = height;
        // }
        sum      += num * height;
        count    += num;

    }

    // if(count == 0)
    //     cout << "cout = 0!!!!!" << h_ptr[h] << endl;
    return sum/count;

    // return max_h;
}

Mat Cloud_Matrix_Loador::rescaleMat(Mat img)
{
    double min_v, max_v;
    minMaxLoc(img, &min_v, &max_v);
    Mat scaled_img = img/max_v;

    Mat ucharMatScaled;
    scaled_img.convertTo(ucharMatScaled, CV_8UC1, 255, 0); 

    return ucharMatScaled;
}

Mat Cloud_Matrix_Loador::get_feature_meanh(Mat img, int size)
{
    Mat blur_sum, blur_count, mean_mat, valid_mask;

    threshold( img, valid_mask, 0, 1, THRESH_BINARY );

    boxFilter(img, blur_sum, img.depth(), Size(size, size), Point(-1,-1), false);
    boxFilter(valid_mask, blur_count, valid_mask.depth(), Size(size, size), Point(-1,-1), false);

    mean_mat = blur_sum/blur_count;

    return mean_mat;
}

Mat Cloud_Matrix_Loador::get_feature_roughness(Mat slope_l, Mat slope_s, Mat h_diff, Mat valid_mask, int size)
{
    Mat slope_diff  = abs(slope_l - slope_s);
    // Mat slope_diff = slope_s;
    Mat diff_sq, roughness_map;
    multiply(slope_diff, slope_diff, diff_sq);

    Mat sq_mean    = get_feature_meanh(diff_sq, size);

    cv::sqrt(abs(sq_mean), roughness_map);

    return roughness_map;
}

void Cloud_Matrix_Loador::get_feature_slope_bycloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt, Mat valid_mask)
{
    /////////////////////// computer normal ///////////////////
    pcl::PointCloud<pcl::Normal>::Ptr       normal_large        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr       normal_small        (new pcl::PointCloud<pcl::Normal>);

    // normal_small = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, S_SLOPE_R);
    // normal_large = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, L_SLOPE_R);

    // float larger_r = map_resolution_ * 5;
    float smaller_r = map_resolution_ * 4;
    if(smaller_r < 0.10)
        smaller_r = 0.10;
    // float larger_r =  smaller_r * 3;

    // normal_large = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, larger_r);
    normal_small = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, smaller_r);
    normal_large = normal_small;

    for(size_t i = 0; i < ground_points_.points.size(); i++)
    {
        if(normal_small->points[i].normal[2] != normal_small->points[i].normal[2] || normal_large->points[i].normal[2] != normal_large->points[i].normal[2])
        // if(normal_small->points[i].normal[2] != normal_small->points[i].normal[2])
            continue;

        int row = ground_points_index_.points[i].y;
        int col = ground_points_index_.points[i].x;
        // slope_map_l_.ptr<float>(row)[col] = 1 - abs(normal_large->points[i].normal[2]);
        slope_map_s_.ptr<float>(row)[col] = 1 - abs(normal_small->points[i].normal[2]);
    }

    slope_map_l_ = get_feature_meanh(slope_map_s_, 5);
}

void Cloud_Matrix_Loador::get_feature_slope_byimgae(Mat img_height, Mat valid_mask)
{
    Mat height_scaled;
    // resize(img_height, height_scaled, Size(), 0.8, 0.8, INTER_NEAREST);

    Mat grad_x_s, grad_y_s, grad_x_l, grad_y_l;
    Mat abs_grad_x, abs_grad_y;

    Scharr(img_height, grad_x_s, img_height.depth(), 1, 0, 3 );
    Scharr(img_height, grad_y_s, img_height.depth(), 0, 1, 3 );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Scharr( img_height, grad_x_l, img_height.depth(), 1, 0, 3 );
    Scharr( img_height, grad_y_l, img_height.depth(), 0, 1, 3 );

    /// Total Gradient (approximate)
    // addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
    Mat sobel_s = (abs(grad_x_s) + abs(grad_y_s))/2;
    Mat sobel_l = (abs(grad_x_l) + abs(grad_y_l))/2;


    // Mat tan_sq, tan_sq1, sin_sq, slope_s, slope_l;
    // multiply(sobel_s, sobel_s, tan_sq);
    // tan_sq1 = tan_sq + 1;
    // sin_sq = tan_sq / tan_sq1;
    // cv::sqrt(sin_sq, slope_map_s_);

    // multiply(sobel_l, sobel_l, tan_sq);
    // tan_sq1 = tan_sq + 1;
    // sin_sq = tan_sq / tan_sq1;
    // cv::sqrt(sin_sq, slope_map_l_);

    slope_map_s_ = sobel_s;
    slope_map_l_ = sobel_l;
    // resize(sobel_l, slope_map_l_, slope_map_l_.size(), 0, 0, INTER_NEAREST);
    slope_map_l_ = get_feature_meanh(slope_map_s_, 5);

    cout << "finish slope" << endl;
}

Mat Cloud_Matrix_Loador::fill_missingvalue(Mat img)
{
    Mat missing_v_mask, temp, missing_v_mat, height_map_close;

    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    morphologyEx(img, height_map_close, MORPH_DILATE, element);

    Mat mean_mat = get_feature_meanh(height_map_close, 3);
    missing_v_mat = get_feature_meanh(mean_mat, 7); 

    threshold( mean_mat, missing_v_mask, 0, 1, THRESH_BINARY_INV );
    multiply(missing_v_mat, missing_v_mask, temp);
    Mat expended = temp + mean_mat;

    return expended;
}

Mat Cloud_Matrix_Loador::get_features(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt)
{
    //////////  assign mean height to missing pixel /////////////////////
    Mat mean_height = fill_missingvalue(height_map_);
    //  Mat mean_height = get_feature_meanh(height_map_, 3);


    /////// compute maximum height difference ////////////////////
    Mat min, max, valid_mask;
    Mat element2 = getStructuringElement(MORPH_RECT, Size(3,3));
    threshold( mean_height, valid_mask, 0, 1, THRESH_BINARY );

    min = (1 - valid_mask) * 5 + mean_height;   // get min value for every pixel
    morphologyEx(min, min, MORPH_ERODE, element2);

    morphologyEx(mean_height, max, MORPH_DILATE, element2);  // get max value for every pixel

    Mat height_diff = (max - min); // get maximum height difference for every pixel
    height_diff_map_ = height_diff.clone();

    //////////  slope  /////////////////////
    // get_feature_slope_byimgae(mean_height, valid_mask);
    get_feature_slope_bycloud(cloud_all_prt, cloud_ground_prt, valid_mask);
    slope_map_l_ = fill_missingvalue(slope_map_l_);
    slope_map_s_ = fill_missingvalue(slope_map_s_);


    ////////// roughness /////////////////////
    Mat roughness_mat = get_feature_roughness(slope_map_l_, slope_map_s_, height_diff, valid_mask, 3);
    roughness_map_ = roughness_mat.clone();
    
    Mat cost_map = compute_cost(height_diff, slope_map_l_, roughness_mat, valid_mask);

    cout << "maximum cost --------  " << max_roughness_ << endl;

    resize(height_diff,   output_height_diff_, Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);
    resize(slope_map_l_,  output_slope_,       Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);
    // resize(roughness_mat, output_roughness_,   Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);
    resize(cost_map,      output_roughness_,   Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);
    resize(mean_height,   output_height_,      Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);
    resize(cost_map,      output_cost_,        Size(map_rows_output_, map_cols_output_), 0, 0, INTER_NEAREST);

    output_height_ = output_height_ - map_height_/2;

    // Mat s_l   = rescaleMat(slope_map_l_);
    // Mat s_s   = rescaleMat(slope_map_s_);
    // Mat m_h   = rescaleMat(output_height_);
    // // // Mat s_s_m = rescaleMat(mean_slope_s);
    // // Mat r     = rescaleMat(cost_map);
    // // Mat d_min = rescaleMat(min);
    // // Mat d_max = rescaleMat(max);
    // Mat d_diff = rescaleMat(height_diff);

    // imshow("mean height", m_h);
    // imshow("s_l", s_l);
    // imshow("s_s", s_s);
    // // // imshow("s_s_m", s_s_m);
    // // // imshow("d_min", d_min);
    // // imshow("d_max", d_max);
    // // imshow("r", r);
    // imshow("d_diff", d_diff);

    // waitKey(50);


    return cost_map;
}

pcl::PointCloud<pcl::PointXYZ> Cloud_Matrix_Loador::get_ground_points(pcl::PointCloud<pcl::PointXYZ> *cloud)
{
    ground_points_.points.clear();
    ground_points_index_.points.clear();

    Mat img_processed;

    for(int row = 0; row < map_rows_; row ++)
    {
        for(int col = 0; col < map_cols_; col ++)
        {   
            mat_ptr     = cloud_mat_.ptr<float>(row, col);

            for(int h = 0; h < map_hs_; h++)
            {
                if(mat_ptr[h] == 0)
                    continue; 
                // Mat subImg = cloud_mat_(cv::Range(0, 100), cv::Range(0, 100));
                // cout << countNonZero(*mat_ptr) << endl;
                bool is_ground = is_potential_ground(mat_ptr, h);

                if(is_ground)
                {
                    float height = get_vertical_mean_height(mat_ptr, h) * map_h_resolution_;
                    height_map_.ptr<float>(row)[col] = height;

                    pcl::PointXYZ point, point_index;
                    point.x = (map_cols_ - col) * map_resolution_ - map_width_/2;
                    point.y = row * map_resolution_ - map_broad_/2;
                    point.z = h * map_h_resolution_ - map_height_/2;
                    
                    point_index.x = col;
                    point_index.y = row;
                    ground_points_.points.push_back(point);
                    ground_points_index_.points.push_back(point_index);
                    break;
                }
            }  
        }
    }

    return ground_points_;
}
