from ultralytics import YOLO
import cv2
import glob
import numpy as np
from pathlib import Path
import open3d as o3d



#intrinsic matrix
K = np.array([
    [2058.72664, 0, 960],
    [0, 2058.72664, 560],
    [0, 0, 1]
])





current_dir = Path(__file__).resolve().parent
car_pics_a_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraA'
car_pics_b_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraB'
car_lidar_a_dir = current_dir / 'Fusion_event-main' / 'data' / 'LidarA'
car_lidar_b_dir = current_dir / 'Fusion_event-main' / 'data' / 'LidarB'

lidar_files = sorted(glob.glob(str(car_lidar_a_dir / '*.ply')))
camera_files = sorted(glob.glob(str(car_pics_a_dir / '*.png')))

model = YOLO("yolov8n.pt")


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


def visualize_results(pcd, bounding_boxes):
    geometries = [pcd]

    for bbox in bounding_boxes:
        geometries.append(bbox)

    o3d.visualization.draw_geometries(geometries)


def load_lidar_data_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def process_lidar_and_camera_data(lidar_files, camera_files):
    for lidar_file, camera_file in zip(lidar_files, camera_files):
        lidar_data = load_lidar_data_ply(lidar_file)

        filtered_lidar = voxel_grid_filter(lidar_data, voxel_size=0.1)
        cleaned_lidar = remove_outliers(filtered_lidar)

        clusters = segment_point_cloud(cleaned_lidar)

        bounding_boxes = create_bounding_boxes(clusters)

        visualize_results(lidar_data, bounding_boxes)


process_lidar_and_camera_data(lidar_files, camera_files)
