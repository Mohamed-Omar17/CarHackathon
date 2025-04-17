from ultralytics import YOLO
import cv2
import numpy as np
import open3d as o3d

# Initialize YOLO model
model = YOLO("yolov8n.pt")

average_object_widths = {
    "car": 2,
    "person": 0.4,
    "truck": 2.5,
    "bus": 2.8,
    "bicycle": 0.6,
    "motorcycle": 0.6,
    # Add more if needed
}




def load_lidar_data_ply(file_path):
    """ Load LiDAR data from a PLY file using Open3D """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def rotate_points_x(points, theta_x):
    """ Rotate points around the X-axis by angle theta_x (radians). """
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    return np.dot(points, rotation_matrix_x.T)


def calculate_depth(focal_length, real_world_width, image_width, bounding_box_width):
    """ Calculate depth using the real-world width and bounding box width """
    return (real_world_width * focal_length) / bounding_box_width


def convert_to_3d_coordinates(u, v, depth, K):
    """ Convert 2D image coordinates to 3D world coordinates using depth and camera matrix """
    inv_K = np.linalg.inv(K)
    pixel_coords = np.array([u, v, 1])
    camera_coords = depth * inv_K @ pixel_coords
    return camera_coords


def project_lidar_to_camera(lidar_points, camera_image, K, theta_x, model):
    """ Rotate LiDAR points, transform, and project them onto the camera image. """
    # Rotate the LiDAR points around the X-axis
    lidar_points = rotate_points_x(lidar_points, theta_x)

    # Transformation matrix from LiDAR to Camera coordinates (mock example)
    T_lidar_to_camera = np.array([
        [0, -1, 0, 0.5],  # Adjust based on your setup
        [0, 0, -1, 0.2],
        [1, 0, 0, 0.3],
        [0, 0, 0, 1]
    ])

    # Convert LiDAR points to homogeneous coordinates
    homogeneous_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    transformed_points = (T_lidar_to_camera @ homogeneous_points.T).T[:, :3]

    # Only keep points in front of the camera (z > 0)
    mask = transformed_points[:, 2] > 0
    transformed_points = transformed_points[mask]

    # Project using intrinsic matrix (camera matrix)
    projected = (K @ transformed_points.T).T
    u = projected[:, 0] / projected[:, 2]  # x / z
    v = projected[:, 1] / projected[:, 2]  # y / z

    # Run YOLO on the image to get bounding boxes
    results = model(camera_image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Get class IDs
    confidences = results.boxes.conf.cpu().numpy()  # Confidence values
    class_names = results.names  # Object class names

    # Camera parameters
    focal_length = K[0, 0]  # Focal length is usually the value in K[0, 0] for most cameras
    # Loop through the bounding boxes and calculate depth for each object
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        u_center = (x1 + x2) / 2
        v_center = (y1 + y2) / 2

        # Get class info
        class_id = class_ids[idx]
        object_type = class_names[class_id] if class_id < len(class_names) else "unknown"
        object_type_lower = object_type.lower()
        real_world_width = average_object_widths.get(object_type_lower, 1.0)

        # Mask projected LiDAR points that fall inside the current bounding box
        in_box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_box_depths = transformed_points[in_box_mask][:, 2]

        # Use LiDAR depth if points are found inside the box
        if len(in_box_depths) > 0:
            depth = np.median(in_box_depths)
            depth_source = "LiDAR"
        else:
            # Fall back to width-based estimate
            bounding_box_width = x2 - x1
            if bounding_box_width == 0:
                continue  # avoid division by zero
            depth = calculate_depth(focal_length, real_world_width, camera_image.shape[1], bounding_box_width)
            depth_source = "YOLO"

        camera_coords = convert_to_3d_coordinates(u_center, v_center, depth, K)

        # Annotate image
        print(f"Object: {object_type}, Depth: {depth:.2f} meters (via {depth_source})")
        cv2.rectangle(camera_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"{object_type} {depth:.2f}m ({depth_source})"
        cv2.putText(camera_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return camera_image


# Function to process both cars (A and B)
def process_car_data(car_number):
    """ Process the data for the given car number (A or B). """
    # Define the rotation angle in radians (for top-to-bottom rotation)
    theta_x = np.radians(10)  # Rotate 10 degrees around the X-axis

    # Define file paths based on car number
    lidar_file_path = f"Fusion_event-main/data/Lidar{car_number}/{car_number}_001.ply"  # Update with your actual file path
    camera_image_path = f"Fusion_event-main/data/Camera{car_number}/{car_number}_001.png"  # Update with your actual file path

    # Load LiDAR data from the PLY file
    lidar_points = load_lidar_data_ply(lidar_file_path)

    # Load camera image
    camera_image = cv2.imread(camera_image_path)

    # Camera intrinsic matrix (example values)
    K = np.array([
        [2058.72664, 0, 960],
        [0, 2058.72664, 560],
        [0, 0, 1]
    ])

    # Project LiDAR points onto camera image with rotation
    camera_image_with_points = project_lidar_to_camera(lidar_points, camera_image, K, theta_x, model)

    # Display the image with projected points
    cv2.imshow(f"Camera Image with Projected LiDAR Points - Car {car_number}", camera_image_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Process Car A and Car B
process_car_data('A')  # For Car A
process_car_data('B')  # For Car B
