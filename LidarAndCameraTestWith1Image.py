import numpy as np
import cv2
import open3d as o3d
import os
from ultralytics import YOLO

# ---- Load and prepare LiDAR ----
def load_lidar_data_ply(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] LiDAR file not found: {file_path}")
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

# ---- Rotate around the X-axis ----
def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

# ---- Visualize 3D point cloud ----
def show_point_cloud(points):
    print("[INFO] Rendering 3D LiDAR point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    z_vals = np.asarray(points)[:, 2]
    colors = np.zeros_like(points)
    colors[:, 2] = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

# ---- Project and label LiDAR points on camera image ----
def project_lidar_to_camera(lidar_points, camera_image, K, yolo_boxes=None):
    homogeneous_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # (N, 4)
    projected_points = K @ homogeneous_points[:, :3].T  # (3, N)

    u = projected_points[0, :] / projected_points[2, :]
    v = projected_points[1, :] / projected_points[2, :]
    distances = np.linalg.norm(rotated_lidar_points[:, [0, 2]], axis=1)

    u = np.clip(u, 0, camera_image.shape[1] - 1)
    v = np.clip(v, 0, camera_image.shape[0] - 1)

    for i in range(len(u)):
        x = int(u[i])
        y = int(v[i])
        d = distances[i]
        point_inside_box = False

        if yolo_boxes:
            for x1, y1, x2, y2 in yolo_boxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    point_inside_box = True
                    break

        if point_inside_box:
            cv2.circle(camera_image, (x, y), 2, (0, 255, 0), -1)  # Green point
            cv2.putText(camera_image, f"{d:.1f}m", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(camera_image, (x, y), 2, (0, 0, 255), -1)  # Red point

    return camera_image

# ---- Main Setup ----
K = np.array([
    [2058.72664, 0, 960],
    [0, 2058.72664, 560],
    [0, 0, 1]
])

lidar_file_path = "Fusion_event-main/data/LidarA/A_001.ply"
camera_image_path = "Fusion_event-main/data/CameraA/A_001.png"

lidar_points = load_lidar_data_ply(lidar_file_path)
camera_image = cv2.imread(camera_image_path)

if lidar_points is not None and camera_image is not None:
    # ---- Rotate LiDAR BEFORE anything else ----
    theta_rad = np.radians(360 - 90.15917205810548)
    R = rotation_matrix_x(theta_rad)
    rotated_lidar_points = lidar_points @ R.T

    camera_offset = np.array([0, 0, 0])
    translated_points = rotated_lidar_points + camera_offset

    # ---- Filter points within distance range ----
    min_range = 1.0  # meters
    max_range = 60.0  # meters
    distances = np.linalg.norm(translated_points, axis=1)
    valid_indices = np.where((distances >= min_range) & (distances <= max_range))
    filtered_lidar_points = translated_points[valid_indices]

    # ---- Load YOLO and detect objects ----
    model = YOLO("yolov8n.pt")
    results = model(camera_image)[0]
    yolo_boxes = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        yolo_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(camera_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # ---- Project points and show distances ----
    camera_image_with_objects = project_lidar_to_camera(
        filtered_lidar_points, camera_image, K, yolo_boxes
    )

    show_point_cloud(filtered_lidar_points)
    cv2.imshow("LiDAR Projected on Image", camera_image_with_objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[ERROR] Missing LiDAR or camera image.")