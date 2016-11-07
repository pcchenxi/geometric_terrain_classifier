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

#define L_SLOPE_R  0.20
#define S_SLOPE_R  0.15

float max_cost_ = 0;

class Point_Matrix
{
    int matrix_size_;

    int matrix_rows_;
    int matrix_cols_;
    int matrix_hights_;

    public:
    float ***mat;
    float ***slope_large;
    float ***slope_small;

    Point_Matrix(int matrix_rows, int matrix_cols, int matrix_hights)
    {
        mat = new float **[matrix_rows];
        slope_large = new float **[matrix_rows];
        slope_small = new float **[matrix_rows];

        for(int row = 0; row <= matrix_rows; row ++)
        {
            mat[row] = new float* [matrix_cols];
            slope_large[row] = new float* [matrix_cols];
            slope_small[row] = new float* [matrix_cols];

            for(int col = 0; col <= matrix_cols; col ++)
            {   
                mat[row][col] = new float[matrix_hights];
                slope_large[row][col] = new float[matrix_hights];
                slope_small[row][col] = new float[matrix_hights];
            }
        }

        matrix_rows_ = matrix_rows;
        matrix_cols_ = matrix_cols;
        matrix_hights_ = matrix_hights;
        matrix_size_ = matrix_rows * matrix_cols * matrix_hights;
    };

    ~Point_Matrix()
    {
        for(int row = 0; row <= matrix_rows_; row ++)
        {
            for(int col = 0; col <= matrix_cols_; col ++)
            {   
                delete [] mat[row][col];
                delete [] slope_large[row][col];
                delete [] slope_small[row][col];
            }
        }
    };

    void clear_matrix()
    {
        for(int row = 0; row <= matrix_rows_; row ++)
        {
            for(int col = 0; col <= matrix_cols_; col ++)
            {   
                memset(mat[row][col], 0, sizeof(float)*matrix_hights_ );
                memset(slope_large[row][col], 0, sizeof(float)*matrix_hights_ );
                memset(slope_small[row][col], 0, sizeof(float)*matrix_hights_ );
            }
        }
    };

};

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

  public:
    Mat     output_height_diff_, output_slope_, output_roughness_;

    // Point_Matrix *cloud_mat_; // 3D feature space

    const float* mat_ptr;
    const float* slope_l_ptr;
    const float* slope_s_ptr;

    pcl::PointCloud<pcl::PointXYZ> ground_points_;
    pcl::PointCloud<pcl::PointXYZ> ground_points_index_;
    pcl::PointCloud<pcl::PointXYZ> cloud_mat_index_;

    //   Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resalution, float map_h_resolution);
      Cloud_Matrix_Loador();
      ~Cloud_Matrix_Loador();

      void init_params(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution);

      pcl::PointCloud<pcl::PointXYZRGB> load_cloud     (pcl::PointCloud<pcl::PointXYZ> cloud, 
                                                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution);
      pcl::PointCloud<pcl::PointXYZ> cloud_filter   (pcl::PointCloud<pcl::PointXYZ> cloud);
      pcl::PointCloud<pcl::PointXYZRGB> reformCloud (pcl::PointCloud<pcl::PointXYZ> cloud, Mat cost_map);
      pcl::PointCloud<pcl::Normal>::Ptr calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                         pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                         float searchRadius );
     pcl::PointCloud<pcl::PointXYZ> cloud_downsample(pcl::PointCloud<pcl::PointXYZ> cloud);                                               

      //////////////////////////////////////////////////
      pcl::PointCloud<pcl::PointXYZ> get_ground_points(pcl::PointCloud<pcl::PointXYZ> *cloud);
      Mat get_costmap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt);
      bool is_potential_ground                      (const float* h_ptr, int h);
      float get_vertical_mean_height                (const float* h_ptr, int h);
      Mat   rescaleMat                              (Mat img);


      ///////////////////////////////////////////// freture
      Mat get_feature_meanh                         (Mat img, Mat valid_mask, int size);
      Mat get_feature_roughness                     (Mat slope_l, Mat slope_s, Mat h_diff, Mat valid_mask, int size);
      Mat compute_cost                              (Mat h_diff, Mat slope, Mat roughness);
};


// Cloud_Matrix_Loador::Cloud_Matrix_Loador(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
Cloud_Matrix_Loador::Cloud_Matrix_Loador()
{
    // map_width_          = map_width;
    // map_broad_          = map_broad;
    // map_height_         = map_height;
    // map_resolution_     = map_resolution;
    // map_h_resolution_   = map_h_resolution;

    // map_rows_           = map_width_/map_resolution_;
    // map_cols_           = map_broad_/map_resolution_;
    // map_hs_             = map_height_/map_h_resolution;

}

Cloud_Matrix_Loador::~Cloud_Matrix_Loador()
{

}

void Cloud_Matrix_Loador::init_params(float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
{
    map_width_          = map_width;
    map_broad_          = map_broad;
    map_height_         = map_height;
    map_resolution_     = map_resolution/3;
    map_h_resolution_   = map_h_resolution;

    map_rows_           = map_width_/map_resolution_;
    map_cols_           = map_broad_/map_resolution_;
    map_hs_             = map_height_/map_h_resolution;
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

        float cost = cost_map.ptr<float>(row)[col];

        float cost_obs = 1;
        float cost_rough = 0.15;
        if(cost >= cost_obs)
        {
            cloud_color.points[i].r = 200;
        }    
        else if(cost > cost_rough)
        {
            cloud_color.points[i].r = 100.0;
            cloud_color.points[i].g = 255.0;
            cloud_color.points[i].b = 0.0;
        }   
        else
        {
            cloud_color.points[i].r = 0.0;
            cloud_color.points[i].g = 0.0;
            cloud_color.points[i].b = 255.0;
        }  
    }

    return cloud_color;
}

pcl::PointCloud<pcl::PointXYZRGB> Cloud_Matrix_Loador::load_cloud(pcl::PointCloud<pcl::PointXYZ> cloud, 
                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution)
{
    init_params(map_width, map_broad, map_height, map_resolution, map_h_resolution);

    int sz[]  = {map_rows_, map_cols_, map_hs_};

    height_map_     = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_l_    = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    slope_map_s_    = Mat(map_rows_, map_cols_, CV_32FC1,  Scalar(0));
    
    cloud_mat_      = Mat(3, sz, CV_32FC1,  Scalar(0.0));
    slope_mat_l_    = Mat(3, sz, CV_32FC1,  Scalar(0.0));
    slope_mat_s_    = Mat(3, sz, CV_32FC1,  Scalar(0.0));

    // cloud_mat_->clear_matrix();

    /////////////////////// filter cloud //////////////////////
    pcl::PointCloud<pcl::PointXYZ> cloud_cut      = cloud_filter(cloud);
    pcl::PointCloud<pcl::PointXYZ> cloud_filtered = cloud_downsample(cloud_cut);

    cout << "before init index space " << cloud_cut.points.size() << endl;
    cloud_mat_index_.points.resize(cloud_cut.points.size());

    cout << "before loading points to mat" << endl;
    for(size_t i = 0; i < cloud_cut.points.size(); i=i+2)
    {
        pcl::PointXYZ point = cloud_cut.points[i];

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

        // cloud_mat_->mat[row][col][hit] += 1;
        // cloud_mat_->slope_large[row][col][hit] = normal_large->points[i].normal[2];
        // cloud_mat_->slope_small[row][col][hit] = normal_small->points[i].normal[2];
        
        cloud_mat_.ptr<float>(row, col)[hit] = i;
        cloud_mat_index_.points[i].x = col;
        cloud_mat_index_.points[i].y = row;
        cloud_mat_index_.points[i].z = hit;
        // cout << i << " " << cloud_mat_.at<float>(row, col, hit) << endl;

    }
    cout << "------------------ Mat initialized " << endl;
    ground_points_ = get_ground_points(&cloud_cut);
    cout << "------------------ ground point initialized  " << ground_points_.size() << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prt           (new pcl::PointCloud<pcl::PointXYZ>(cloud_filtered));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt    (new pcl::PointCloud<pcl::PointXYZ>(ground_points_));
    Mat cost_map = get_costmap(cloud_prt, cloud_ground_prt);
    cout << "------------------got cost map " << ground_points_.size() << endl;

    pcl::PointCloud<pcl::PointXYZRGB> cloud_color = reformCloud(ground_points_, cost_map);
    cout << "------------------finished reformCloud " << cloud_color.points.size() << endl;

    // pcl::PointCloud<pcl::PointXYZRGB> cloud_color;
    // return cloud_color;
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
    int max_num = 0;
    int max_h = 0;
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

Mat Cloud_Matrix_Loador::get_feature_roughness(Mat slope_l, Mat slope_s, Mat h_diff, Mat valid_mask, int size)
{
    max_cost_ = 0;
    // Mat slope_diff  = abs(slope_l - slope_s);
    Mat slope_diff = slope_s;
    Mat diff_sq, roughness_map;
    multiply(slope_diff, slope_diff, diff_sq);

    Mat sq_mean    = get_feature_meanh(diff_sq, valid_mask, size);

    cv::sqrt(sq_mean, roughness_map);

    return roughness_map;
}

Mat Cloud_Matrix_Loador::compute_cost(Mat h_diff, Mat slope, Mat roughness)
{
    Mat cost_map = Mat(h_diff.rows, h_diff.cols, CV_32FC1,  Scalar(0));
    for(int row = 0; row < h_diff.rows; row ++)
    {
        for(int col = 0; col < h_diff.cols; col ++)
        {
            float height_diff   = h_diff.ptr<float>(row)[col];
            float slope_v       = slope.ptr<float>(row)[col];
            float roughness_v   = roughness.ptr<float>(row)[col];

            cost_map.at<float>(row, col) = roughness_v;

            if(height_diff > 0.4 || slope_v > 0.5)
            {
                cost_map.at<float>(row, col) = 1.0;
            }    
            
            else if(roughness_v > max_cost_)
                max_cost_ = roughness_v;
        }
    }

    return cost_map;
}


Mat Cloud_Matrix_Loador::get_costmap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_prt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt)
{
    /////////////////////// computer normal ///////////////////
    pcl::PointCloud<pcl::Normal>::Ptr       normal_large        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr       normal_small        (new pcl::PointCloud<pcl::Normal>);

    float larger_r = map_resolution_ * 4;
    float smaller_r = larger_r / 2;
    normal_large = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, larger_r);
    normal_small = calculateSurfaceNormal(cloud_ground_prt, cloud_all_prt, smaller_r);

    cout << "       ------------------ finish ground points normal computation" << endl;
    for(size_t i = 0; i < ground_points_.points.size(); i++)
    {
        if(normal_small->points[i].normal[2] != normal_small->points[i].normal[2] || normal_large->points[i].normal[2] != normal_large->points[i].normal[2])
        // if(normal_small->points[i].normal[2] != normal_small->points[i].normal[2])
            continue;

        int row = ground_points_index_.points[i].y;
        int col = ground_points_index_.points[i].x;
        slope_map_l_.ptr<float>(row)[col] = 1 - abs(normal_large->points[i].normal[2]);
        slope_map_s_.ptr<float>(row)[col] = 1 - abs(normal_small->points[i].normal[2]);
    }

    cout << "       ------------------ loaded normal" << endl;

    /////// compute maximum height difference ////////////////////
    Mat min, max, valid_mask;
    Mat element2 = getStructuringElement(MORPH_RECT, Size(3,3));
    threshold( height_map_, valid_mask, 0, 1, THRESH_BINARY );
    
    min = (1 - valid_mask) * 5 + height_map_;   // get min value for every pixel
    morphologyEx(min, min, MORPH_ERODE, element2);

    morphologyEx(height_map_, max, MORPH_DILATE, element2);  // get max value for every pixel

    Mat height_diff = (max - min) & valid_mask; // get maximum height difference for every pixel

    // Mat mean_mat = get_feature_meanh(height_map_, valid_mask, 5);
    Mat roughness_mat = get_feature_roughness(slope_map_l_, slope_map_s_, height_diff, valid_mask, 3);

    Mat cost_map = compute_cost(height_diff, slope_map_l_, roughness_mat);

    resize(height_diff,   output_height_diff_, Size(), 0.33, 0.33, INTER_NEAREST);
    resize(slope_map_l_,  output_slope_,       Size(), 0.33, 0.33, INTER_NEAREST);
    resize(roughness_mat, output_roughness_,   Size(), 0.33, 0.33, INTER_NEAREST);

    // Mat s_l   = rescaleMat(slope_map_l_);
    // Mat s_s   = rescaleMat(slope_map_s_);
    // // Mat s_s_m = rescaleMat(mean_slope_s);
    // Mat r     = rescaleMat(roughness_scaleddown);
    // Mat d_min = rescaleMat(min);
    // Mat d_max = rescaleMat(max);
    // Mat d_diff = rescaleMat(diff);

    // // imshow("mean height", mean_mat);
    // imshow("s_l", s_l);
    // imshow("s_s", s_s);
    // // imshow("s_s_m", s_s_m);
    // // imshow("d_min", d_min);
    // // imshow("d_max", d_max);
    // imshow("diff", diff);
    // imshow("roughness_mat", r);

    // waitKey(50);


    return cost_map;
}

pcl::PointCloud<pcl::PointXYZ> Cloud_Matrix_Loador::get_ground_points(pcl::PointCloud<pcl::PointXYZ> *cloud)
{
    cout << "in set_ground_points" << endl;
    ground_points_.points.clear();
    ground_points_index_.points.clear();

    Mat img_processed;
    float scale = map_h_resolution_;

    for(int row = 0; row < map_rows_; row ++)
    {
        for(int col = 0; col < map_cols_; col ++)
        {   
            mat_ptr     = cloud_mat_.ptr<float>(row, col);

            for(int h = 0; h < map_hs_; h ++)
            {
                bool is_ground = is_potential_ground(mat_ptr, h);

                if(is_ground)
                {
                    // float height = get_vertical_mean_height(mat_ptr, h) * scale;
                    int cloud_index = mat_ptr[h];
                    height_map_.at<float>(row, col) = h * map_h_resolution_;

                    pcl::PointXYZ point, point_index;
                    point.x = (map_cols_ - col) * map_resolution_ - map_width_/2;
                    point.y = row * map_resolution_ - map_broad_/2;
                    point.z = h * map_h_resolution_ - map_height_/2;
                    
                    point_index.x = col;
                    point_index.y = row;
                    ground_points_.points.push_back(point);
                    ground_points_index_.points.push_back(point_index);
                    continue;
                }
            }  
        }
    }

    cout << "finish ground point slelction" << endl;

    return ground_points_;
}
