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

#define MIN_REQUIRED_H  2
#define GROUND_ACCURACY 0.05

#define L_SLOPE_R  0.35
#define S_SLOPE_R  0.20

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

    Mat     ground_mask_;
    Mat     slope_mat_l_, slope_mat_s_, cloud_mat_; // 3D point space
    Mat     slope_map_l_, slope_map_s_, height_map_; // 2D feature space

    vector<int> index_col_;
    vector<int> index_row_;

    const float* mat_ptr;
    const float* slope_l_ptr;
    const float* slope_s_ptr;


  public:
      Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resalution, float map_h_resolution);
      ~Cloud_Matrix_Loador();

      Mat* get_cloud_matrix();
      pcl::PointCloud<pcl::PointXYZRGB> load_cloud     (pcl::PointCloud<pcl::PointXYZ> cloud);
      pcl::PointCloud<pcl::PointXYZ> cloud_filter   (pcl::PointCloud<pcl::PointXYZ> cloud);
      pcl::PointCloud<pcl::PointXYZRGB> reformCloud (pcl::PointCloud<pcl::PointXYZ> cloud, Mat cost_map);
      pcl::PointCloud<pcl::Normal>::Ptr calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                         pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                         float searchRadius );
     pcl::PointCloud<pcl::PointXYZ> cloud_downsample(pcl::PointCloud<pcl::PointXYZ> cloud);                                               

      //////////////////////////////////////////////////
      Mat set_ground_points                         ();
      bool is_potential_ground                      (const float* h_ptr, int h);
      float get_vertical_mean_height                (const float* h_ptr, int h);
      Mat   rescaleMat                              (Mat img);


      ///////////////////////////////////////////// freture
      Mat get_feature_meanh                         (Mat img, Mat valid_mask, int size);
      Mat get_feature_roughness                     (Mat slope_l, Mat slope_s, Mat valid_mask, int size);
};


Cloud_Matrix_Loador::Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
{
    map_width_          = map_width;
    map_broad_          = map_broad;
    map_height_         = map_height;
    map_resolution_     = map_resolution;
    map_h_resolution_   = map_h_resolution;

    map_rows_           = map_width_/map_resolution_;
    map_cols_           = map_broad_/map_resolution_;
    map_hs_             = map_height_/map_h_resolution;

    // cloud_mat_     = Mat(3, sz, CV_8UC1,  Scalar(0));
    // ground_mask_   = Mat(3, sz, CV_8UC1,  Scalar(0));
    // height_map_    = Mat(map_rows_, map_cols_, CV_8UC1,  Scalar(0));
}

Cloud_Matrix_Loador::~Cloud_Matrix_Loador()
{

}

Mat* Cloud_Matrix_Loador::get_cloud_matrix()
{
    return &cloud_mat_;
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

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_passthrough);
    sor.setLeafSize (0.01, 0.01, 0.01);
    sor.filter (*cloud_passthrough);
    cout << "after voxel filter  " << cloud_passthrough->points.size() << endl;

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
    sor.setLeafSize (map_resolution_, map_resolution_, map_resolution_);
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
        int row = index_row_[i];
        int col = index_col_[i];

        float cost = cost_map.at<float>(row, col);

        cloud_color.points[i].r = 255.0 * cost;
    }

    return cloud_color;
}

pcl::PointCloud<pcl::PointXYZRGB> Cloud_Matrix_Loador::load_cloud(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    const int sz[]  = {map_rows_, map_cols_, map_hs_};

    height_map_     = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_l_    = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_s_    = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));

    cloud_mat_      = Mat(3, sz, CV_32FC1,  Scalar(0));
    slope_mat_l_    = Mat(3, sz, CV_32FC1,  Scalar(0));
    slope_mat_s_    = Mat(3, sz, CV_32FC1,  Scalar(0));

    /////////////////////// filter cloud //////////////////////
    pcl::PointCloud<pcl::PointXYZ> cloud_cut      = cloud_filter(cloud);
    pcl::PointCloud<pcl::PointXYZ> cloud_filtered = cloud_downsample(cloud_cut);

    /////////////////////// computer normal ///////////////////
    pcl::PointCloud<pcl::Normal>::Ptr       normal_large        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr       normal_small        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr     cloud_prt           (new pcl::PointCloud<pcl::PointXYZ>(cloud_filtered));
    pcl::PointCloud<pcl::PointXYZ>::Ptr     cloud_all_prt       (new pcl::PointCloud<pcl::PointXYZ>(cloud_filtered));

    normal_large = calculateSurfaceNormal(cloud_prt, cloud_all_prt, L_SLOPE_R);
    normal_small = calculateSurfaceNormal(cloud_prt, cloud_all_prt, S_SLOPE_R);

    index_col_.clear();
    index_row_.clear();
    index_col_.resize(cloud_filtered.points.size());
    index_row_.resize(cloud_filtered.points.size());

    for(size_t i = 0; i < cloud_filtered.points.size(); i++)
    {
        if(normal_small->points[i].normal[2] != normal_small->points[i].normal[2])
            continue;

        pcl::PointXYZ point = cloud_filtered.points[i];

        point.x     += map_width_/2;
        point.y     += map_broad_/2;
        point.z     += map_height_/2;

        // cout << "start" << endl;
        int col     = point.x / map_resolution_;
        int row     = point.y / map_resolution_;
        int hit     = point.z / map_h_resolution_;

        col         = map_cols_ - col;

        // cout << row << " " << col << " " << hit << " " << (int)cloud_mat.at<uchar>(row, col, hit) << endl;
        if(hit > map_hs_ || row > map_rows_ || col > map_cols_ || hit < 0 || row < 0 || col < 0)
            continue;

        cloud_mat_.at<float>(row, col, hit) += 1;
        slope_mat_l_.at<float>(row, col, hit) = normal_large->points[i].normal[2];
        slope_mat_s_.at<float>(row, col, hit) = normal_small->points[i].normal[2];
        
        index_col_[i] = col;
        index_row_[i] = row;
    }

    Mat cost_map = set_ground_points();
    cout << "finished init" << endl;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_color = reformCloud(cloud_filtered, cost_map);
    cout << "finished reformCloud " << cloud_color.points.size() << endl;

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_color;
    return cloud_color;
}



bool Cloud_Matrix_Loador::is_potential_ground(const float* h_ptr, int h)
{
    int min_required_step   = MIN_REQUIRED_H / map_h_resolution_;
    int empty_count         = 0;

    bool is_ground          = false;

    float base_n            = h_ptr[h];
    if(base_n == 0)
        return is_ground;

    for(size_t height = h+1; height < map_hs_; height ++)
    {
        float num = h_ptr[height];
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

    return is_ground;
}

float Cloud_Matrix_Loador::get_vertical_mean_height(const float* h_ptr, int h)
{
    int sum = 0;
    int count = 0;
    int mean_step = GROUND_ACCURACY / map_h_resolution_;

    int end_index = h - mean_step;
    if(end_index  < 0)
        end_index = 0;
        
    for(size_t height = h; height >= end_index; height --)
    {
        float num = h_ptr[height];
        if(num == 0)
            continue;
        
        sum      += num * height;
        count    += num;

    }


    return sum/count;
}

Mat Cloud_Matrix_Loador::rescaleMat(Mat img)
{
    double min_v, max_v;
    minMaxLoc(img, &min_v, &max_v);
    img = img/max_v;

    Mat ucharMatScaled;
    img.convertTo(ucharMatScaled, CV_8UC1, 255, 0); 

    return ucharMatScaled;
}

Mat Cloud_Matrix_Loador::get_feature_meanh(Mat img, Mat valid_mask, int size)
{
    Mat blur_sum, blur_count, mean_mat;
    boxFilter(img, blur_sum, img.depth(), Size(size, size), Point(-1,-1), false);
    boxFilter(valid_mask, blur_count, valid_mask.depth(), Size(size, size), Point(-1,-1), false);

    mean_mat = blur_sum/blur_count;
    return mean_mat;
}

Mat Cloud_Matrix_Loador::get_feature_roughness(Mat slope_l, Mat slope_s, Mat valid_mask, int size)
{
    // Mat slope_diff  = abs(slope_l - slope_s);
    Mat diff_sq, roughness;
    multiply(slope_s, slope_s, diff_sq);
    Mat sq_mean     = get_feature_meanh(diff_sq, valid_mask, 5);

    cv::sqrt(sq_mean, roughness);

    return sq_mean;
}

Mat Cloud_Matrix_Loador::set_ground_points()
{
    Mat img_processed;
    // float scale = 255/map_height_ * map_h_resolution_;
    float scale = map_h_resolution_;

    for(size_t row = 0; row < map_rows_; row ++)
    {
        for(size_t col = 0; col < map_cols_; col ++)
        {   
            mat_ptr     = cloud_mat_.ptr<float>(row, col);
            slope_l_ptr = slope_mat_l_.ptr<float>(row, col);
            slope_s_ptr = slope_mat_s_.ptr<float>(row, col);

            // cout << row << " " << map_rows_ << endl;
            // cout << col << " " << map_cols_ << endl;
            for(size_t h = 0; h < map_hs_; h ++)
            {
                // uchar a = mat_ptr[h];
                // uchar a = cloud_mat_.at<uchar>(row, col, h);
                bool is_ground = is_potential_ground(mat_ptr, h);

                if(is_ground)
                {
                    // int height = h * scale ;
                    float height = get_vertical_mean_height(mat_ptr, h) * scale;
                    height_map_.at<float>(row, col) = height;
                    slope_map_l_.at<float>(row, col) = 1 - abs(slope_l_ptr[h]);
                    slope_map_s_.at<float>(row, col) = 1 - abs(slope_s_ptr[h]);

                        // cout << slope_s_ptr[h] << endl;
                }
            }   
        }
    }

    /// fill in the holes
    // Mat element = getStructuringElement(MORPH_RECT, Size(7,7));
    // Mat element2 = getStructuringElement(MORPH_RECT, Size(3,3));

    Mat min, max, diff, valid_mask, valid_mask_nav;
    threshold( height_map_, valid_mask, 0, 1, THRESH_BINARY );
    // valid_mask_nav = ~valid_mask;

    Mat mean_mat = get_feature_meanh(height_map_, valid_mask, 5);
        cout << "mean_mat" << endl;
    Mat roughness_mat = get_feature_roughness(slope_map_l_, slope_map_s_, valid_mask, 5);
        cout << "roughness_mat" << endl;
    // diff         = abs(mean_mat - height_map_);
    // diff         = diff & valid_mask;

    // // morphologyEx(height_map_, img_processed, MORPH_CLOSE, element);

    // morphologyEx(height_map_, max, MORPH_DILATE, element2);
    // morphologyEx(height_map_, min, MORPH_ERODE, element2);

    // diff = height_map_ - min;
    // img_processed = (diff == height_map_);
    // // img_processed = diff & valid_mask;

    // Mat h = rescaleMat(height_map_);
    // Mat m = rescaleMat(mean_mat);
    // Mat d = rescaleMat(diff);

    // Mat s_l = rescaleMat(slope_map_l_);
    // Mat s_s = rescaleMat(slope_map_s_);
    Mat r = rescaleMat(roughness_mat);

    // imshow("height", h);
    // imshow("s_l", s_l);
    // imshow("s_s", s_s);
    imshow("roughness_mat", r);

    waitKey(10);

    return roughness_mat;
}