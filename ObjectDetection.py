from ultralytics import YOLO
import cv2
import glob
import numpy as np
from pathlib import Path
import open3d as o3d

# Intrinsic Matrix
K = np.array([
    [2058.72664, 0, 960],
    [0, 2058.72664, 560],
    [0, 0, 1]
])

# Directories for your data
current_dir = Path(__file__).resolve().parent
car_pics_a_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraA'
car_pics_b_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraB'
car_lidar_a_dir = current_dir / 'Fusion_event-main' / 'data' / 'LidarA'
car_lidar_b_dir = current_dir / 'Fusion_event-main' / 'data' / 'LidarB'

lidar_files = sorted(glob.glob(str(car_lidar_a_dir / '*.ply')))
camera_files = sorted(glob.glob(str(car_pics_a_dir / '*.png')))

model = YOLO("yolov8n.pt")

# LiDAR data processing functions (same as before)
def segment_point_cloud(pcd, eps=0.2, min_points=10):
    points = np.asarray(pcd.points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    clusters = []
    for label in np.unique(labels):
        if label == -1:
            continue  # Ignore noise points (label == -1)
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    return clusters

def create_bounding_boxes(clusters):
    bounding_boxes = []
    for cluster in clusters:
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bounding_boxes.append(bbox)
    return bounding_boxes

def voxel_grid_filter(pcd, voxel_size):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def visualize_results(camera_image_with_points):
    # Simply display the camera image with projected points
    cv2.imshow("Camera Image with Projected LiDAR Points", camera_image_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_lidar_data_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# New function to project LiDAR points onto camera image
def project_lidar_to_camera(lidar_points, camera_image, K):
    # Convert LiDAR points to homogeneous coordinates (N x 4)
    homogeneous_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # (N, 4)

    # Project the 3D points onto the 2D image plane using the intrinsic matrix
    # K is (3, 3), and homogeneous_points.T is (4, N)
    # We need to multiply K (3, 3) with homogeneous_points.T (4, N)
    # The result will be a 3 x N matrix, which we transpose to get (N, 3)
    projected_points = K @ homogeneous_points[:, :3].T  # Multiply intrinsic matrix by (3, N)

    # Normalize by z to get 2D pixel coordinates (u, v)
    u = projected_points[0, :] / projected_points[2, :]  # x / z
    v = projected_points[1, :] / projected_points[2, :]  # y / z

    # Clip the points to stay within image bounds
    u = np.clip(u, 0, camera_image.shape[1] - 1)
    v = np.clip(v, 0, camera_image.shape[0] - 1)

    # Draw the projected points onto the camera image (for visualization)
    for i in range(len(u)):
        cv2.circle(camera_image, (int(u[i]), int(v[i])), 2, (0, 0, 255), -1)  # Red dots

    return camera_image


# Main function to process both LiDAR and camera data
def process_lidar_and_camera_data(lidar_files, camera_files):
    for lidar_file, camera_file in zip(lidar_files, camera_files):
        lidar_data = load_lidar_data_ply(lidar_file)

        filtered_lidar = voxel_grid_filter(lidar_data, voxel_size=0.1)
        cleaned_lidar = remove_outliers(filtered_lidar)

        clusters = segment_point_cloud(cleaned_lidar)
        bounding_boxes = create_bounding_boxes(clusters)

        # Load camera image
        camera_image = cv2.imread(camera_file)

        # Project LiDAR points onto the camera image
        lidar_points = np.asarray(lidar_data.points)
        camera_image_with_points = project_lidar_to_camera(lidar_points, camera_image, K)

        # Visualize the 2D result (camera image with projected points)
        visualize_results(camera_image_with_points)
# Run the data processing
process_lidar_and_camera_data(lidar_files, camera_files)
