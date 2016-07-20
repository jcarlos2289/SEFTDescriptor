/*
procesor receive roscaffe_msg with 
information about localization of objetcs

  Created on: jul 4 , 2016
 *      Author: jcarlos2289

*/
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "roscaffe_msgs/LocalizedPredictions.h"
#include "roscaffe_msgs/Prediction.h"
#include "cloud_clusterer/cluster_clouds.h"
#include <ros/ros.h>
#include "ros/package.h"
#include <geometry_msgs/Point.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <visualization_msgs/Marker.h>
#include <cmath>
#include <stdexcept> 
#include <eigen3/Eigen/Dense>

//#include <pcl/visualization/pcl_visualizer.h>


ros::Subscriber sub;
ros::Subscriber cluster_sub;
ros::Publisher pub;
ros::Publisher pub_rviz;
ros::Publisher vis_cent_pub;
ros::Publisher vis_tag_pub;
ros::Publisher tag_pub;
ros::Publisher pub_cluster_rviz;
ros::Publisher pub_reader_results;

std::vector<double> global_confidences ;
std::vector<std::string> global_labels ;
std::vector<geometry_msgs::Point> global_position  ;
  
std::vector<geometry_msgs::Point> global_centroids_position;
std::vector<pcl::PointCloud<pcl::PointXYZ> > global_clusterVector;
pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud;

int global_predictionNum, global_locationsNum, global_tagsAmount;

typedef struct {
    double r,g,b;
} COLOUR;

COLOUR GetColour(double v,double vmin,double vmax)
{
   COLOUR c = {1.0,1.0,1.0}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }

   return(c);
}

double CalculateEuclideanDistance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{	
	double diffx = p1.x - p2.x;
	double diffy = p1.y - p2.y;
    double diffz = p1.z - p2.z;
	double diffx_sqr = pow(diffx,2);
	double diffy_sqr = pow(diffy,2);
    double diffz_sqr = pow(diffz,2);
	double distance = sqrt (diffx_sqr + diffy_sqr +diffz_sqr);

return distance;
}

std::string printXYZ(geometry_msgs::Point point){
    std::stringstream ss;
    
    ss << point.x <<", " << point.y <<", " <<point.z ;
    return ss.str();
}

float mahalanobisDistance(geometry_msgs::Point pointData,  pcl::PointCloud<pcl::PointXYZ> clusterData){
    
    int numPoints= clusterData.points.size();
    Eigen:: MatrixXf point(3,1);
    point << pointData.x,pointData.y,pointData.z; 
    //media
    geometry_msgs::Point mean;
    
    mean.x =0;
    mean.y =0;
    mean.z =0;
    
    for(int i = 0; i< numPoints; ++i){
        mean.x += clusterData.points.at(i).x/numPoints;
        mean.y += clusterData.points.at(i).y/numPoints;
        mean.z += clusterData.points.at(i).z/numPoints;
     }

    
    //varianza
    geometry_msgs::Point variance;
    
    variance.x =0;
    variance.y =0;
    variance.z =0;
    
    
     for(int i = 0; i< numPoints; ++i){
        variance.x += pow(clusterData.points.at(i).x - mean.x,2)/(numPoints-1);
        variance.y += pow(clusterData.points.at(i).y - mean.y,2)/(numPoints-1);
        variance.z += pow(clusterData.points.at(i).z - mean.z,2)/(numPoints-1);
     }
    
    //Standard Deviation
    geometry_msgs::Point sd;
    
    sd.x = sqrt(variance.x);
    sd.y = sqrt(variance.y);
    sd.z = sqrt(variance.z);
        
    //covariance 
    float cvXY = 0;
    float cvXZ = 0;
    float cvYZ = 0;
    
    for(int i =0 ; i< numPoints; ++i){
        cvXY += ((clusterData.points.at(i).x - mean.x) * (clusterData.points.at(i).y - mean.y)) / numPoints;
        cvXZ += ((clusterData.points.at(i).x - mean.x) * (clusterData.points.at(i).z - mean.z)) / numPoints;
        cvYZ += ((clusterData.points.at(i).y - mean.y) * (clusterData.points.at(i).z - mean.z)) / numPoints;
     }
        
    Eigen::Matrix3f covarianceMatrix ;
    covarianceMatrix  << variance.x , cvXY , cvXZ 
                        ,cvXY , variance.y , cvYZ 
                        ,cvXZ , cvYZ , variance.z ;
    
      
    Eigen::Matrix3f inverseMatrix ;
    inverseMatrix = covarianceMatrix.inverse();
    
    Eigen:: MatrixXf pointTranspose(3,1);
    Eigen:: MatrixXf meanVector(3,1);
      
    meanVector << mean.x, mean.y, mean.z;
   
    pointTranspose=point -meanVector;
    pointTranspose.transposeInPlace();
             
    Eigen:: MatrixXf maha;
    maha  = (pointTranspose * inverseMatrix * (point -meanVector));
    
    float mahalanobis = sqrt(maha.coeff(0,0));
    
    return mahalanobis;
    
}




void findNearPoints(){
   std::vector<std::pair<int, int> > pairs; //pairs of index (indexPosLabel, indexPosCentroid)
   std::stringstream ss, dist; 
   
       //vizualizar los tags
     uint32_t shape = visualization_msgs::Marker::SPHERE;
     
     visualization_msgs::Marker marker, markerCent;
     marker.header.frame_id = "camera_rgb_optical_frame";
     marker.header.stamp = ros::Time();
     
     markerCent.header.frame_id = "camera_rgb_optical_frame";
     markerCent.header.stamp = ros::Time();
         
    
    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "tags_space";
    markerCent.ns = "cent_space";
   
    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = 9;//shape; //9-> text
    //marker.text = global_labels.at(0);
    
    markerCent.type =9;// shape; //9-> text
    //markerCent.text = global_labels.at(0);
    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    markerCent.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
   
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    markerCent.pose.orientation.x = 0.0;
    markerCent.pose.orientation.y = 0.0;
    markerCent.pose.orientation.z = 0.0;
    markerCent.pose.orientation.w = 1.0;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.01;
    
    markerCent.scale.x = 0.05;
    markerCent.scale.y = 0.05;
    markerCent.scale.z = 0.01;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;
    
    markerCent.color.r = 143.0f;
    markerCent.color.g = 143.0f;
    markerCent.color.b = 143.0f;
    markerCent.color.a = 1.0;

    marker.lifetime = ros::Duration();
    markerCent.lifetime = ros::Duration();
    
    pcl::PointCloud<pcl::PointXYZRGBA> cloud_ram;
    pcl::PointXYZRGBA point_ram;
   
   int yID = 0;
  ss <<"Indentified Clusters in the Cloud: " <<global_clusterVector.size() << std::endl;
  ss << "\nIdentified pairs\n" <<std::endl;
   for(int i = 0; i < global_position.size() ; ++i ){  //
       for(int j = 0; j < global_centroids_position.size() ; ++j  ){
           float maha = mahalanobisDistance(global_position.at(i),global_clusterVector.at(j));
           dist << "Mahalanobis Distance\t\t" << maha << std::endl;
        if (maha<1.5){
            
            //CalculateEuclideanDistance(global_position.at(i), global_centroids_position.at(j))< 0.05){  // EuclideanDistance
            std::stringstream g ;
            g << j ;
            
            pairs.push_back(std::make_pair(i,j));
             marker.scale.z = 0.01;
             marker.id = yID;
            // numero del centroide
            marker.pose.position.x = global_centroids_position.at(j).x;//
            marker.pose.position.y = global_centroids_position.at(j).y;//
            marker.pose.position.z = global_centroids_position.at(j).z; //
            marker.text = g.str();//global_labels.at(i);
            vis_tag_pub.publish(marker);
            ++yID;
            
            
           /* 
            //position tags
            marker.id = yID;
            marker.color.g = 1.0f; //position en verde
            marker.pose.position.x = global_position.at(i).x;  //centroids_position.at(0).x;//;
            marker.pose.position.y = global_position.at(i).y ;//+0.005*j;  // centroids_position.at(0).y ;//
            marker.pose.position.z = global_position.at(i).z; //centroids_position.at(0).z; //
            marker.text = global_labels.at(i);
            vis_tag_pub.publish(marker);
           // marker.color.g = 0.0f;
            
            ++yID;
            
            //centroides
            markerCent.id = yID;
            // centroide en azul
            markerCent.pose.position.x = global_centroids_position.at(j).x;//
            markerCent.pose.position.y = global_centroids_position.at(j).y;//
            markerCent.pose.position.z = global_centroids_position.at(j).z; //
            markerCent.text = g.str();//global_labels.at(i);
            vis_cent_pub.publish(markerCent); 
            ++yID;*/
            
            //Cloud con puntos en las coordenadas
            
            //position roscaffe en amarillo  rgb(255,217,40)
            point_ram.x = global_position.at(i).x;
            point_ram.y = global_position.at(i).y;
            point_ram.z = global_position.at(i).z;
            point_ram.r = 255;
            point_ram.g = 217;
            point_ram.b = 40;
            point_ram.a = 255;
           
            cloud_ram.points.push_back(point_ram);
            // std::cout << "Linea 190 "<<std::endl;
             
            //position centroide en violeta 
            point_ram.x = global_centroids_position.at(j).x;
            point_ram.y = global_centroids_position.at(j).y;
            point_ram.z = global_centroids_position.at(j).z;
            point_ram.r = 253;
            point_ram.g = 20;
            point_ram.b = 255;
            point_ram.a = 255;
            
            cloud_ram.points.push_back(point_ram);
                        
            ss <<"LabelIndex\t\t" << i
                <<"\nLabel\t\t\t" <<global_labels.at(i)  
                <<"\nConfidence\t\t" <<global_confidences.at(i)
                <<"\nPosition\t\t" << printXYZ(global_position.at(i)) 
                <<"\nCentroidIndex\t\t"<< j 
                <<"\nCentroidXYZ\t\t" << printXYZ(global_centroids_position.at(j)) 
                <<"\n"
                <<std::endl;
                             
                      
       //  try {
        //ss << global_labels.at(i)  <<"\t\t\t" << printXYZ(global_position.at(i)) <<"\t\t\t\t\t" <<j << "\t\t" << printXYZ(global_centroids_position.at(j)) <<std::endl;
     /*   } // global_labels.at(i)
  catch (const std::out_of_range& oor) {
    std::cout <<"Error aqui i = " << i <<" "<< oor.what() << std::endl;
  }*/
        
        }//fin if
       } 
   }
  //  vis_cent_pub.publish(markerCent); 
    //vis_tag_pub.publish(marker);
    
    //pairs of index (indexPosLabel, indexPosCentroid)
    //.first-> index del Label 
    //.second-> index del centroide
    
 int idCount = yID+1;   
 
 markerCent.color.r=143;
 markerCent.color.b=143;
 markerCent.color.g=143;
 
 markerCent.scale.z = 0.01;
 
 for(int h = 0; h < global_centroids_position.size(); ++h){
    
     std::stringstream tagList;
     tagList << h;
    
     // centroide en gris
    markerCent.pose.position.x = global_centroids_position.at(h).x;//
    markerCent.pose.position.y = global_centroids_position.at(h).y;//
    markerCent.pose.position.z = global_centroids_position.at(h).z; //
   
    for(int i = 0; i < pairs.size(); ++i){
         if(pairs.at(i).second == h){
              tagList <<"\n" <<global_confidences.at(pairs.at(i).first) <<"\t" <<global_labels.at(pairs.at(i).first);     
           }
             
       }   
       
         //marker.id = idCount;
        // ++idCount;
         markerCent.text = tagList.str();//global_labels.at(i);
         markerCent.id = idCount;
         ++idCount;
         //vis_tag_pub.publish(marker);
         vis_cent_pub.publish(markerCent);
        
           
    }
     
   
   sensor_msgs::PointCloud2 cloud_msg;
   pcl::toROSMsg(cloud_ram,cloud_msg);
   cloud_msg.header.frame_id="camera_rgb_optical_frame"; //cloud_processed
   tag_pub.publish(cloud_msg);
    
   ss<< "\nPairs found\t\t" << pairs.size() << std::endl;
   std::cout << ss.str();

   std_msgs::String resultsTopic;
   resultsTopic.data = ss.str();

   //pub_reader_results.publish(resultsTopic);

   //Imprimir Mahalanobis Distance Values
// std::cout << dist.str();
      //-------------------Makers------------------------
   ros::shutdown();
}


void prediction_cb (roscaffe_msgs::LocalizedPredictions input)
{
  sub.shutdown ();
  ROS_INFO("Message received from /roscaffe/localized_predictions");
  
  int predictionNum, locationsNum, tagsAmount;
  std::vector<double> confidences  = input.confidence;
  std::vector<std::string> labels = input.label;
  std::vector<geometry_msgs::Point> position = input.position;
  std::stringstream sd;
  
  std::vector<double> confidencesSelected;
  std::vector<std::string> labelsSelected;
  std::vector<geometry_msgs::Point> positionSelected;
  
  
  for(int i = 0; i< confidences.size(); ++i){
      if(confidences.at(i) > 0.01){
          confidencesSelected.push_back(confidences.at(i));
          labelsSelected.push_back(labels.at(i));
          positionSelected.push_back(position.at(i));
      }
  }
  tagsAmount = confidencesSelected.size();
  
  sd << "Selected Labels " <<  labelsSelected.size() << std::endl;
  for( int j = 0; j<labelsSelected.size(); ++j ){
      sd <<" index = " << j << " " << labelsSelected.at(j) << std::endl;
  }
  
  std::cout <<sd.str();
  predictionNum = input.n;
  locationsNum = input.m;
  
  std::stringstream ss;
  ss<< "Objects Detected: " << locationsNum <<std::endl;
  ss<< "Locations Detected: " << position.size() <<std::endl;
  ss<< "Labels Selected: " << tagsAmount <<std::endl;
 
  
 /* 
  for(int j = 0; j< labels.size(); j+= predictionNum){
    ss <<"\n#" << (j+1) <<" -> XYZ = " << position.at(j).x <<" _ " << position.at(j).y <<" _ " << position.at(j).z << std::endl;
   for (int i = 0; i<predictionNum; ++i)
    {
         ss <<"\t" << confidences.at(j+i)  << " _ " <<  labels.at(j+i) << std::endl; 
    }
   }
  */
  
  std::cout << ss.str() <<std::endl;
  
  global_confidences.swap(confidencesSelected);
  global_labels.swap(labelsSelected);
  global_position.swap(positionSelected);
  global_predictionNum = predictionNum;
  global_tagsAmount = tagsAmount;
  global_locationsNum = locationsNum;
  
  //ros::shutdown();
}

void cluster_cb (cloud_clusterer::cluster_clouds input){
     ROS_INFO("Message received from /cluster_data_topic");
     std::stringstream ss;
     
     int clusterNum = input.n;
     std::vector<geometry_msgs::Point> centroids_position = input.centroids;
     
     //obtengo la nube en formato de pcl
    pcl::PCLPointCloud2 pc2_cloud;
    pcl_conversions::toPCL(input.cloud, pc2_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pc2_cloud,*cloud);


    std::vector<pcl::PointCloud<pcl::PointXYZ> > clusterVector;
   
    for ( int i = 0; i < input.clusters.size(); ++i){
        pcl::PCLPointCloud2 pc2_cloud_ram;
        pcl_conversions::toPCL(input.clusters.at(i), pc2_cloud_ram);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ram (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pc2_cloud_ram,*cloud_ram);
        clusterVector.push_back(*cloud_ram);
        
    }
        
     
     ss<< "Number of clusters detected:  " << input.clusters.size() << std::endl;
     std::cout << ss.str();
     
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colorCloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_total (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
     
            
     copyPointCloud(*cloud, *colorCloud);
     double max = clusterVector.size() +1;
     
     COLOUR col = GetColour(1.0,1.0/max,max/max);
     for(int i =0; i<colorCloud->points.size(); ++i)
     {
         colorCloud->points.at(i).r = col.r*255;
         colorCloud->points.at(i).g = col.g*255;
         colorCloud->points.at(i).b = col.b*255;
         colorCloud->points.at(i).a = 100;
     }
     
    *cloud_total = *colorCloud;
     
     //double max = clusterVector.size();
     for(int i = 0; i< clusterVector.size(); ++i){
          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cluster_ram (new pcl::PointCloud<pcl::PointXYZRGBA>);
         copyPointCloud(clusterVector.at(i), *cluster_ram);
         
         COLOUR col2 = GetColour(i/max,1.0/max,max/max);
          for(int j =0; j<cluster_ram->points.size(); ++j)
            {
                cluster_ram->points.at(j).r = col2.r*255;
                cluster_ram->points.at(j).g = col2.g*255;
                cluster_ram->points.at(j).b = col2.b*255;
                cluster_ram->points.at(j).a = 255;
            }
         
         if(i != 0)   
            *cloud_cluster +=  *cluster_ram;
           else
           *cloud_cluster =  *cluster_ram;
     }
      
      
     // *cloud_total -= *colorCloud;
      
     sensor_msgs::PointCloud2 cloud_msg;
     pcl::toROSMsg(*cloud_total,cloud_msg);
     pub_rviz.publish(cloud_msg);
     
     sensor_msgs::PointCloud2 cloud_cluster_msg;
     pcl::toROSMsg(*cloud_cluster,cloud_cluster_msg);
     cloud_cluster_msg.header.frame_id="camera_rgb_optical_frame";
     
     pub_cluster_rviz.publish(cloud_cluster_msg);
     
     global_centroids_position.swap(centroids_position);
     global_clusterVector.swap(clusterVector);
     global_cloud = cloud;
     
     findNearPoints();
     
}



int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "msg_reader");
  boost::shared_ptr<ros::NodeHandle> nh_ptr_;
  
  nh_ptr_ = boost::make_shared<ros::NodeHandle> ();

  // Create a ROS subscriber for the input point cloud
  // topicName, queueSize, callbackFunction 
  
  sub = nh_ptr_->subscribe ("/roscaffe/localized_predictions", 1, prediction_cb);
  ROS_INFO("Waiting for data from  /roscaffe/localized_predictions.");
  cluster_sub = nh_ptr_->subscribe("/cluster_data_topic", 1, cluster_cb);
  pub_rviz = nh_ptr_->advertise<sensor_msgs::PointCloud2>("/cloud_processed",1);
  vis_cent_pub = nh_ptr_->advertise<visualization_msgs::Marker>( "cent_marker", 0 );
  vis_tag_pub = nh_ptr_->advertise<visualization_msgs::Marker>( "tag_marker", 0 );
  tag_pub  = nh_ptr_->advertise<sensor_msgs::PointCloud2>("/tags_location",0);
  pub_cluster_rviz = nh_ptr_->advertise<sensor_msgs::PointCloud2>("/clusters_processed",0);
  pub_reader_results= nh_ptr_->advertise<std_msgs::String>("/cluster_tag_location",0);
 
  

  // Create a ROS publisher for the output point cloud
  //pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
   ros::spin();
  /*while(nh.ok()){
     
  }
  
  ros::spin ();*/
}

