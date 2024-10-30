#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360, generate_dust3r_path
from scene.cameras import PseudoCamera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        print(f"Checking dataset type with source path: {args.source_path}")
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Recognized COLMAP dataset structure in sparse/ folder.")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views)
        
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        
        elif os.path.exists(os.path.join(args.source_path, "sparse/0/images.txt")) and os.path.exists(os.path.join(args.source_path, "sparse/0/cameras.txt")):
            print("Found images.txt and cameras.txt in sparse/0, assuming DUST3R data set!")
            # Dust3r 경로 전달
            scene_info = sceneLoadTypeCallbacks["Dust3r"](os.path.join(args.source_path, "sparse/0"), args.images, args.eval, args.n_views)
        
        else:
            print("Error: Could not recognize scene type. Dataset structure may not match expectations.")
            print(f"Expected COLMAP structure: {os.path.join(args.source_path, 'sparse')}")
            print(f"Expected Blender file: {os.path.join(args.source_path, 'transforms_train.json')}")
            print(f"Expected Dust3r structure: {os.path.join(args.source_path, 'sparse/0/images.txt')} and {os.path.join(args.source_path, 'sparse/0/cameras.txt')}")
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            if args.source_path.find('llff'):
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('360'):
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            view = self.train_cameras[resolution_scale][0]
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]