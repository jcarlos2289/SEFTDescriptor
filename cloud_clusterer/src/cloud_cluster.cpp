#include <ros/ros.h>
#include "ros/package.h"
#include <geometry_msgs/Point.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include "cloud_clusterer/cluster_clouds.h"




ros::Publisher cluster_pub_;
ros::Subscriber rgbdimage_sub_;
boost::shared_ptr<ros::NodeHandle> nh_ptr_;


// Single RGB-D image processing
void rgbdImageCallback (const sensor_msgs::PointCloud2ConstPtr& in_cloud)
{
  ROS_INFO ("Received point cloud from roscaffe.");
  //rgbdimage_sub_.shutdown();
  // Convert Point Cloud
  pcl::PCLPointCloud2 pc2_cloud;
  pcl_conversions::toPCL(*in_cloud, pc2_cloud);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pc2_cloud,*cloud);

  // Convert RGB image
  sensor_msgs::ImagePtr in_img (new sensor_msgs::Image);
  pcl::toROSMsg(*in_cloud, *in_img);
  cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(in_img, sensor_msgs::image_encodings::BGR8);
  
  //--------------------------Clustering procedure---------------------------------
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);


std::vector< pcl::PointCloud<pcl::PointXYZ> > clustersDetected;

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clustersDetected.push_back(*cloud_cluster);
    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "/home/jcarlos2289/Documentos/clustererClouds/cloud_cluster_" << j << ".pcd";
    
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //se guarda la nube en la carpeta
    j++;
  }
 std::stringstream sl;
 sl<< "Cluster Detected: " << clustersDetected.size() <<std::endl;
  std::cout << "Clustering Procedure finished !!!! " << std::endl;
  std::cout <<sl.str();
  
 //------------------------------end Clustering Procedure------------------------ 
 
 //----------------------------------centroid calculation------------------------
 
 // Create and accumulate points

std::vector<pcl::PointXYZ> centroids;
std::vector< sensor_msgs::PointCloud2> clusterVector;
for ( std::vector< pcl::PointCloud<pcl::PointXYZ> >::iterator it  = clustersDetected.begin(); it!= clustersDetected.end(); ++it){
    
    pcl::CentroidPoint<pcl::PointXYZ> centroid;
        
    for ( int i = 0; i < it->points.size(); ++i){
        centroid.add(
            pcl::PointXYZ (it->points[i].x,
                           it->points[i].y,
                           it->points[i].z)
                    );
    
    }
    
    pcl::PointXYZ c1;
    centroid.get (c1);
    centroids.push_back(c1);
    
   sensor_msgs::PointCloud2 in_img;
   pcl::toROSMsg(*it, in_img);
   clusterVector.push_back(in_img);
    
}
 std::stringstream sla;
 sla<< "Centroids Detected: " << centroids.size() <<std::endl;


std::vector<geometry_msgs::Point> positions;

 
  for (int i = 0; i < centroids.size(); ++i){
    sla << "\nCluster #: " << i  << "\n" 
        << " "    << centroids.at(i).x
        << " "    << centroids.at(i).y
        << " "    << centroids.at(i).z << std::endl;
        geometry_msgs::Point p;
        p.x = centroids.at(i).x;
        p.y = centroids.at(i).y;
        p.z = centroids.at(i).z;
        positions.push_back (p);
        }
 
std::cout << sla.str() <<std::endl;
 
 //------------------------------------------------------------------------------
 //----------------------Sending results-----------------------------------------
 //ingresar todos los datos al objeto msg(cuertMSG)  y luego enviarlo
  //sensor_msgs::PointCloud2 msg;
  
  //cluster_pub_.publish (msg);
  
  cloud_clusterer::cluster_clouds msg;
  
  
  msg.header.seq = in_cloud->header.seq;
  msg.header.stamp = in_cloud->header.stamp;
  msg.header.frame_id = in_cloud->header.frame_id;
  msg.n = centroids.size();
  msg.cloud = *in_cloud;
  msg.centroids = positions;
  msg.clusters = clusterVector;
  
  cluster_pub_.publish (msg);
  ros::shutdown();
  
  
  
 
 
 //------------------------------------------------------------------------------
 
  
 
}

void connectCallback (const ros::SingleSubscriberPublisher &pub)
{
 /* if (prediction_pub_.getNumSubscribers() == 1)
  {
    // Subscribe to camera
    rgbdimage_sub_ = nh_ptr_->subscribe<sensor_msgs::PointCloud2> ("depth_registered", 1, rgbdImageCallback);//1
    ROS_INFO ("Starting predictions stream.");
  }*/
}

void disconnectCallback (const ros::SingleSubscriberPublisher &pub)
{
  /*if (prediction_pub_.getNumSubscribers() == 0)
  {
    // Unsubscribe to camera
    rgbdimage_sub_.shutdown ();
    ROS_INFO ("Stopping predictions stream.");
  }*/
}

int main(int argc, char** argv)
{
 
  // ROS init
  ros::init(argc, argv, "cloud_clusterer");
  nh_ptr_ = boost::make_shared<ros::NodeHandle> ();
  
   
  // Advertise prediction service for batch processing
  cluster_pub_ = nh_ptr_->advertise<cloud_clusterer::cluster_clouds> ("/cluster_data_topic", 10);
  
  // Subscribe to RGB-D image (moved to connect callback)
  rgbdimage_sub_ = nh_ptr_->subscribe<sensor_msgs::PointCloud2> ("/camera/cloud/points", 1, rgbdImageCallback);
  ROS_INFO("Waiting for point cloud.");
  // Return control to ROS
  ros::spin();
  
  return 0;
}
