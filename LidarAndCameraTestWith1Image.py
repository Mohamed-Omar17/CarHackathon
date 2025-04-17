from ultralytics import YOLO
import cv2
import numpy as np
import open3d as o3d

# Initialize YOLO model
model = YOLO("yolov8n.pt")

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

def project_lidar_to_camera(lidar_points, camera_image, K, theta_x, model):
    """ Rotate LiDAR points, transform, and project them onto the camera image. """
    # Rotate the LiDAR points around the X-axis
    lidar_points = rotate_points_x(lidar_points, theta_x)

    # Transformation matrix from LiDAR to Camera coordinates (mock example)
    T_lidar_to_camera = np.array([
        [0, -1, 0, 0.5],   # This should be adjusted based on your setup
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

    # Debug: Print the first few projected points
    print("First few projected points (u, v):")
    print(list(zip(u[:5], v[:5])))

    # Run YOLO on the image to get bounding boxes
    results = model(camera_image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)

    # Draw points: green if inside detected object, red otherwise
    for i in range(len(u)):
        px, py = int(u[i]), int(v[i])

        # Check if the point is within image boundaries
        if px < 0 or px >= camera_image.shape[1] or py < 0 or py >= camera_image.shape[0]:
            continue  # Skip if point is outside the image

        color = (0, 0, 255)  # Default: red

        # Check if the point is inside any bounding box
        for (x1, y1, x2, y2) in boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                color = (0, 255, 0)  # Inside object: green
                break

        # Draw point on image
        cv2.circle(camera_image, (px, py), 2, color, -1)

    # Draw YOLO bounding boxes on the image
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(camera_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

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
