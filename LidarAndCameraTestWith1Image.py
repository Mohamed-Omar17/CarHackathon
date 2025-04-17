from ultralytics import YOLO
import cv2
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import json

# === CONFIGURABLE PARAMETERS ===
MIN_HEIGHT = 0
MAX_HEIGHT = 5
MAX_DISTANCE = 30  # Max distance in meters

# === Load YOLO model ===
model = YOLO("yolov8n.pt")

# === Predefined values ===
average_object_sizes = {
    "car": {"width": 2},
    "person": {"height": 1.7},
    "truck": {"width": 2.5},
    "bus": {"width": 2.8},
    "bicycle": {"width": 0.6},
    "motorcycle": {"width": 0.6},
}

# === Camera intrinsics ===
K = np.array([
    [2058.72664, 0, 960],
    [0, 2058.72664, 560],
    [0, 0, 1]
])


# === Utility functions ===
def load_lidar_data_ply(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] LiDAR file not found: {file_path}")
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def calculate_depth_estimate(focal_length, real_size, image_size):
    return (real_size * focal_length) / image_size


def convert_to_3d_coordinates(u, v, depth, K):
    inv_K = np.linalg.inv(K)
    pixel_coords = np.array([u, v, 1])
    camera_coords = depth * inv_K @ pixel_coords
    return camera_coords


def scale_bounding_box(x1, y1, x2, y2, scale_factor=1.7):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    new_width = width * scale_factor
    new_height = height * scale_factor
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    return new_x1, new_y1, new_x2, new_y2


def rotate_and_translate(local_points, car_pos, car_rot_deg):
    # Convert rotation from degrees to radians
    theta = np.radians(car_rot_deg - 90)

    # Create rotation matrix for the car's rotation
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Apply rotation and translation (car's position)
    rotated_points = [car_pos + R @ np.array([x, z]) for x, z in local_points]

    return rotated_points


def estimate_object_distances(camera_image, K, model):
    focal_length = K[0, 0]
    results = model(camera_image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = results.names

    local_object_coords = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        object_type = class_names[class_ids[idx]]
        u_center = (x1 + x2) / 2
        v_center = (y1 + y2) / 2

        if object_type == "person":
            bbox_size = y2 - y1
            real_size = average_object_sizes.get("person", {}).get("height", 1.7)
        else:
            bbox_size = x2 - x1
            real_size = average_object_sizes.get(object_type, {}).get("width", 2)

        if bbox_size == 0:
            continue

        depth = calculate_depth_estimate(focal_length, real_size, bbox_size)
        coords = convert_to_3d_coordinates(u_center, v_center, depth, K)
        x_pos, z_pos = coords[0], coords[2]

        local_object_coords.append((x_pos, z_pos))

        cv2.rectangle(camera_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"{object_type}: X={x_pos:.2f}m, Z={z_pos:.2f}m"
        cv2.putText(camera_image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return camera_image, local_object_coords


def filter_objects_within_distance(objects, car_pos, max_distance):
    filtered_objects = []
    for obj in objects:
        x, z = obj
        dist = np.linalg.norm(np.array([x - car_pos[0], z - car_pos[1]]))
        if dist <= max_distance:
            filtered_objects.append(obj)
    return filtered_objects


# === Plotting ===
all_world_objects = []


def process_car_data(car_number, car_x_pos, car_z_pos, car_rot_deg, color):
    camera_image_path = f"Fusion_event-main/data/Camera{car_number}/{car_number}_001.png"
    print(f"\n[INFO] Loading image for Car {car_number}...")
    camera_image = cv2.imread(camera_image_path)
    if camera_image is None:
        print(f"[ERROR] Could not load image {camera_image_path}")
        return

    estimated_image, local_objects = estimate_object_distances(camera_image, K, model)
    window_name = f"Estimated Distances - Car {car_number}"
    cv2.imshow(window_name, estimated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    car_pos = np.array([car_x_pos, car_z_pos])
    world_coords = rotate_and_translate(local_objects, car_pos, car_rot_deg)

    # Filter objects within MAX_DISTANCE from the car
    filtered_objects = filter_objects_within_distance(world_coords, car_pos, MAX_DISTANCE)
    all_world_objects.extend((coord, color) for coord in filtered_objects)


# === Load and process JSON files ===
json_directory = "Fusion_event-main/data/input"  # Replace with the actual path to your JSON files directory

# Get the list of all JSON files in the directory
json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# Loop through each JSON file to process car data
# Loop through each JSON file to process car data
for json_file in json_files:
    json_file_path = os.path.join(json_directory, json_file)

    with open(json_file_path, 'r') as f:
        car_data = json.load(f)

    x_carA = car_data["CarA_Location"][0]
    z_carA = car_data["CarA_Location"][1]
    rot_carA = car_data["CarA_Rotation"]

    x_carB = car_data["CarB_Location"][0]
    z_carB = car_data["CarB_Location"][1]
    rot_carB = car_data["CarB_Rotation"]

    print(str(x_carA) + " " + str(z_carA) + " " + str(x_carB) + " " + " " + str(z_carB))

    process_car_data('A', x_carA, z_carA, rot_carA, 'red')
    process_car_data('B', x_carB, z_carB, rot_carB, 'blue')

    carA_camera = car_data["CarA_Camera"]
    carB_camera = car_data["CarB_Camera"]
    print(f"Processing data from {json_file} with camera paths: {carA_camera}, {carB_camera}")

    # === Per-file plot ===
    plt.figure(figsize=(10, 8))
    plt.plot(x_carA, z_carA, 'ro', label='Car A')
    plt.plot(x_carB, z_carB, 'bo', label='Car B')
    plt.text(x_carA, z_carA + 1, 'Car A', color='red', ha='center')
    plt.text(x_carB, z_carB + 1, 'Car B', color='blue', ha='center')

    for (x, z), color in all_world_objects:
        plt.plot(x, z, marker='^', color=color)

    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title(f'Car and Object Positions - {json_file}')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Clear world object list for the next file
    all_world_objects.clear()

