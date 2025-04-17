from ultralytics import YOLO
import cv2
import numpy as np
import open3d as o3d
import os

# Initialize YOLO model
model = YOLO("yolov8n.pt")

average_object_widths = {
    "car": 2,
    "person": 0.4,
    "truck": 2.5,
    "bus": 2.8,
    "bicycle": 0.6,
    "motorcycle": 0.6,
}


def load_lidar_data_ply(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] LiDAR file not found: {file_path}")
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def rotate_points_x(points, theta_x):
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    return np.dot(points, rotation_matrix_x.T)


def calculate_depth(focal_length, real_world_width, image_width, bounding_box_width):
    return (real_world_width * focal_length) / bounding_box_width


def convert_to_3d_coordinates(u, v, depth, K):
    inv_K = np.linalg.inv(K)
    pixel_coords = np.array([u, v, 1])
    camera_coords = depth * inv_K @ pixel_coords
    return camera_coords


def project_lidar_to_camera(lidar_points, camera_image, K, theta_x, model):
    lidar_points = rotate_points_x(lidar_points, theta_x)

    T_lidar_to_camera = np.array([
        [0, -1, 0, 0.5],
        [0, 0, -1, 0.2],
        [1, 0, 0, 0.3],
        [0, 0, 0, 1]
    ])

    homogeneous_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    transformed_points = (T_lidar_to_camera @ homogeneous_points.T).T[:, :3]

    mask = transformed_points[:, 2] > 0
    transformed_points = transformed_points[mask]

    projected = (K @ transformed_points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    # Get YOLO detection results
    results = model(camera_image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = results.names

    # Loop through each LiDAR point and check if it's inside a bounding box
    for i in range(len(u)):
        x, y = int(u[i]), int(v[i])

        # Check if the point is within the bounds of the image
        if 0 <= x < camera_image.shape[1] and 0 <= y < camera_image.shape[0]:
            is_inside_box = False
            # Check if the point (u, v) is within any bounding box
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    is_inside_box = True
                    break

            # Draw the LiDAR point
            if is_inside_box:
                cv2.circle(camera_image, (x, y), 1, (0, 255, 0), -1)  # Green for object points
            else:
                cv2.circle(camera_image, (x, y), 1, (0, 0, 255), -1)  # Red for non-object points

    # Highlight detected objects with bounding boxes and distances
    focal_length = K[0, 0]
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        # Calculate the center of the bounding box
        u_center = (x1 + x2) / 2
        v_center = (y1 + y2) / 2

        # Get the class ID and object type
        class_id = class_ids[idx]
        object_type = class_names[class_id] if class_id < len(class_names) else "unknown"

        # Estimate depth from LiDAR or YOLO (if LiDAR data is insufficient)
        in_box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_box_depths = transformed_points[in_box_mask][:, 2]

        if len(in_box_depths) > 0:
            depth = np.median(in_box_depths)
            depth_source = "LiDAR"
        else:
            bounding_box_width = x2 - x1
            if bounding_box_width == 0:
                continue
            depth = calculate_depth(focal_length, 1.0, camera_image.shape[1], bounding_box_width)
            depth_source = "YOLO"

        # Convert to 3D coordinates and extract X, Z
        camera_coords = convert_to_3d_coordinates(u_center, v_center, depth, K)
        x_pos = camera_coords[0]
        z_pos = camera_coords[2]

        # Draw the bounding box and display X, Z distances
        cv2.rectangle(camera_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow box
        pos_label = f"X: {x_pos:.2f}m, Z: {z_pos:.2f}m"
        cv2.putText(camera_image, pos_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Return the image with projected LiDAR points and highlighted objects with distances
    return camera_image


def process_car_data(car_number):
    theta_x = np.radians(10)

    lidar_file_path = f"Fusion_event-main/data/Lidar{car_number}/{car_number}_001.ply"
    camera_image_path = f"Fusion_event-main/data/Camera{car_number}/{car_number}_001.png"

    print(f"\n[INFO] Loading data for Car {car_number}...")
    print(f"[INFO] LiDAR path: {lidar_file_path}")
    print(f"[INFO] Image path: {camera_image_path}")

    lidar_points = load_lidar_data_ply(lidar_file_path)
    if lidar_points is None:
        print(f"[ERROR] Skipping Car {car_number} due to missing LiDAR data.")
        return

    camera_image = cv2.imread(camera_image_path)
    if camera_image is None:
        print(f"[ERROR] Could not load camera image from {camera_image_path}")
        return

    K = np.array([
        [2058.72664, 0, 960],
        [0, 2058.72664, 560],
        [0, 0, 1]
    ])

    camera_image_with_points = project_lidar_to_camera(lidar_points, camera_image, K, theta_x, model)

    window_name = f"Camera Image with Projected LiDAR Points - Car {car_number}"
    cv2.imshow(window_name, camera_image_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run for Car A and B
process_car_data('A')
process_car_data('B')
