# Clusters Centroids in the original cloud

std_msgs/Header header
int8 n                          		# Number of clusters
sensor_msgs/PointCloud2 cloud 	        # Original Cloud
sensor_msgs/PointCloud2[] clusters 		# Detected Clusters
geometry_msgs/Point[] centroids  		# Centroids of the n clusters

