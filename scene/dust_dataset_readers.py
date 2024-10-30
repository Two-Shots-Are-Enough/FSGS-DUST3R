import os
import numpy as np
import open3d as o3d
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.gaussian_model import BasicPointCloud
from collections import namedtuple
from scene.scene_definitions import CameraInfo, SceneInfo

def qvec2rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def parse_images_txt(images_txt_path):
    cam_extrinsics = {}
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            image_id, qvec, tvec, camera_id, image_name = int(parts[0]), list(map(float, parts[1:5])), list(map(float, parts[5:8])), int(parts[8]), parts[9]
            cam_extrinsics[image_name] = {'image_id': image_id, 'qvec': np.array(qvec), 'tvec': np.array(tvec), 'camera_id': camera_id}
    return cam_extrinsics

def parse_cameras_txt(cameras_txt_path):
    cam_intrinsics = {}
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            camera_id, model, width, height, params = int(parts[0]), parts[1], int(parts[2]), int(parts[3]), list(map(float, parts[4:]))
            cam_intrinsics[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cam_intrinsics

def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return BasicPointCloud(points=np.asarray(pcd.points), colors=np.asarray(pcd.colors), normals=np.asarray(pcd.normals))

def compute_bounds_for_camera(cam_info, point_cloud):
    R, T = qvec2rotmat(cam_info['qvec']), cam_info['tvec'].reshape(3, 1)
    points_cam = R @ (point_cloud.points.T - T)
    Z_cam = points_cam[2, :][points_cam[2, :] > 0]
    return np.percentile(Z_cam, 0.1) if Z_cam.size else 0.1, np.percentile(Z_cam, 99.9) if Z_cam.size else 10.0

def readDust3rSceneInfo(path, images_folder, eval, n_views=0, llffhold=8):
    cam_extrinsics = parse_images_txt(os.path.join(path, "images.txt"))
    cam_intrinsics = parse_cameras_txt(os.path.join(path, "cameras.txt"))
    point_cloud = load_point_cloud(os.path.join(path, "points3D.ply"))

    cam_infos, rgb_mapping = [], sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))])
    for idx, image_name in enumerate(sorted(cam_extrinsics.keys())):
        extr, intr = cam_extrinsics[image_name], cam_intrinsics[cam_extrinsics[image_name]['camera_id']]
        focal_x, focal_y, width, height = intr['params'][0], intr['params'][1], intr['width'], intr['height']
        near, far = compute_bounds_for_camera(extr, point_cloud)
        cam_infos.append(CameraInfo(
            uid=extr['image_id'], R=qvec2rotmat(extr['qvec']), T=extr['tvec'],
            FovX=focal2fov(focal_x, width), FovY=focal2fov(focal_y, height),
            image=Image.open(os.path.join(images_folder, rgb_mapping[idx])), image_path=rgb_mapping[idx],
            image_name=image_name, width=width, height=height, mask=None, bounds=np.array([near, far])
        ))

    train_cameras = [c for i, c in enumerate(cam_infos) if i % llffhold] if eval else cam_infos
    return SceneInfo(
        point_cloud=point_cloud, train_cameras=train_cameras[:n_views] if n_views else train_cameras,
        test_cameras=[], nerf_normalization={"translate": np.array([0, 0, 0]), "radius": 1.0},
        ply_path=os.path.join(path, "points3D.ply")
    )
