import numpy as np
import open3d as o3d
import os

def qvec2rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])
    return R

def parse_images_txt(images_txt_path):
    cam_extrinsics = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        cam_extrinsics[image_name] = {
            'image_id': image_id,
            'qvec': np.array([qw, qx, qy, qz]),
            'tvec': np.array([tx, ty, tz]),
            'camera_id': camera_id
        }
    return cam_extrinsics

def parse_cameras_txt(cameras_txt_path):
    cam_intrinsics = {}
    with open(cameras_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))
        cam_intrinsics[camera_id] = {
            'camera_id': camera_id,
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    return cam_intrinsics

def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    return points

def compute_bounds_for_camera(cam_info, point_cloud):
    R = qvec2rotmat(cam_info['qvec'])
    T = cam_info['tvec'].reshape(3, 1)
    
    points_world = point_cloud.T  # (3, N)
    points_cam = R @ (points_world - T)
    
    # Z axis
    Z_cam = points_cam[2, :]
    
    # Selecting the points in front of camera
    Z_cam = Z_cam[Z_cam > 0]
    
    if len(Z_cam) == 0:
        near = 0.1
        far = 10.0
    else:
        near = np.percentile(Z_cam, 0.1)
        far = np.percentile(Z_cam, 99.9)
        near = max(near, 0.1)
        far = far * 1.1
    
    return near, far

def generate_poses_bounds(images_txt_path, cameras_txt_path, ply_path, output_dir):
    cam_extrinsics = parse_images_txt(images_txt_path)
    cam_intrinsics = parse_cameras_txt(cameras_txt_path)
    point_cloud = load_point_cloud(ply_path)
    
    poses_bounds = []

    for image_name in sorted(cam_extrinsics.keys()):
        extr = cam_extrinsics[image_name]
        intr = cam_intrinsics[extr['camera_id']]
        
        R = qvec2rotmat(extr['qvec'])
        T = extr['tvec']
        
        pose = np.hstack([R, T.reshape(3, 1)])
        
        # Calculating Bounds
        near, far = compute_bounds_for_camera(extr, point_cloud)
        
        pose_flat = pose.flatten()
        
        pose_bounds = np.hstack([pose_flat, [near, far]])
        
        poses_bounds.append(pose_bounds)

    poses_bounds = np.array(poses_bounds)
    print("Loaded poses_bounds shape:", poses_bounds.shape)

    output_path = os.path.join(output_dir, 'poses_bounds.npy')
    np.save(output_path, poses_bounds)
    
    print(f"poses_bounds.npy Generated: {output_path}")

generate_poses_bounds(
    images_txt_path="/dataset/dust/bicycle_large/sparse/0/images.txt",
    cameras_txt_path="/dataset/dust/bicycle_large/sparse/0/cameras.txt",
    ply_path="/dataset/dust/bicycle_large/sparse/0/points3D.ply",
    output_dir="/dataset/dust/bicycle_large"
)
