from typing import NamedTuple, List
import numpy as np
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str
