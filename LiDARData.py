import open3d as o3d
from pathlib import Path

current_dir = Path(__file__).resolve().parent
lidar_a_file_path = current_dir / 'Fusion_event-main' / 'data' / 'LidarA'
lidar_b_file_path = current_dir / 'Fusion_event-main' / 'data' / 'LidarB'

def load_lidar_data(file_path):
    # Load the LiDAR point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# Example usage
lidar_file_path = 'path/to/lidar_data.pcd'
pcd = load_lidar_data(lidar_file_path)

# Visualize the LiDAR point cloud
o3d.visualization.draw_geometries([pcd])
