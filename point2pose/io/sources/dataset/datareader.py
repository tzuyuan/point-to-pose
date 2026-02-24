# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# This code is taken from the bundlesdf repo: https://github.com/NVlabs/BundleSDF


import pickle, glob, cv2, imageio, os, trimesh, pdb, logging
import numpy as np
import ruamel.yaml

yaml = ruamel.yaml.YAML()


def depth2xyzmap(depth, K):
    invalid_mask = depth < 0.1
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij")
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)


class YCBInIsaacReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None):
        self.video_dir = video_dir
        self.downscale = downscale
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        self.masks_root = os.path.join(self.video_dir, "masks")
        self.poses_root = os.path.join(self.video_dir, "annotated_poses")

        self.object_names = self._discover_object_names()
        self.mask_files_by_object = {}
        self.gt_pose_files_by_object = {}

        if len(self.object_names) > 0:
            for obj_name in self.object_names:
                self.mask_files_by_object[obj_name] = sorted(
                    glob.glob(os.path.join(self.masks_root, obj_name, "*"))
                )
                self.gt_pose_files_by_object[obj_name] = sorted(
                    glob.glob(os.path.join(self.poses_root, obj_name, "*"))
                )
        else:
            # Backward compatibility: flat single-object layout.
            self.object_names = ["object_0"]
            self.mask_files_by_object["object_0"] = sorted(
                glob.glob(os.path.join(self.masks_root, "*"))
            )
            self.gt_pose_files_by_object["object_0"] = sorted(
                glob.glob(os.path.join(self.poses_root, "*"))
            )

        # Keep legacy member for callers expecting single-object pose files.
        self.gt_pose_files = self.gt_pose_files_by_object[self.object_names[0]]

        # First-frame masks for initialization per object.
        self.init_mask_files_by_object = {}
        for obj_name in self.object_names:
            obj_masks = self.mask_files_by_object.get(obj_name, [])
            self.init_mask_files_by_object[obj_name] = (
                obj_masks[0] if obj_masks else None
            )

        self.videoname_to_object = {
            "cracker_box": "003_cracker_box",
            "mustard_bottle": "006_mustard_bottle",
            "extra_large_clamp": "052_extra_large_clamp",
        }

    def _discover_object_names(self):
        if not os.path.isdir(self.masks_root):
            return []
        object_names = [
            d
            for d in sorted(os.listdir(self.masks_root))
            if os.path.isdir(os.path.join(self.masks_root, d))
        ]
        return object_names

    @property
    def num_objects(self):
        return len(self.object_names)

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def get_object_names(self):
        return list(self.object_names)

    def __len__(self):
        return len(self.color_files)

    def get_gt_pose(self, i, obj_name=None):
        if obj_name is None:
            obj_name = self.object_names[0]

        pose_files = self.gt_pose_files_by_object.get(obj_name, [])
        if len(pose_files) == 0:
            logging.info(f"GT pose files not found for object {obj_name}, return None")
            return None

        idx = min(i, len(pose_files) - 1)
        try:
            pose = np.loadtxt(pose_files[idx]).reshape(4, 4)
            return pose
        except Exception:
            logging.info(
                f"GT pose not found/readable for object {obj_name}, frame {i}, return None"
            )
            return None

    def get_gt_poses(self, i):
        poses = {}
        for obj_name in self.object_names:
            poses[obj_name] = self.get_gt_pose(i, obj_name=obj_name)
        return poses

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        if color.shape[-1] == 4:
            color = color[..., :3]  # Drop alpha channel

        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def _read_and_resize_mask(self, mask_file):
        if mask_file is None or not os.path.exists(mask_file):
            return np.zeros((self.H, self.W), dtype=np.uint8)

        mask = cv2.imread(mask_file, -1)
        if mask is None:
            return np.zeros((self.H, self.W), dtype=np.uint8)
        if len(mask.shape) == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        return mask

    def get_mask(self, i, obj_name=None, use_init_mask=False):
        if obj_name is None:
            obj_name = self.object_names[0]

        if use_init_mask:
            return self._read_and_resize_mask(
                self.init_mask_files_by_object.get(obj_name, None)
            )

        obj_mask_files = self.mask_files_by_object.get(obj_name, [])
        if len(obj_mask_files) == 0:
            return np.zeros((self.H, self.W), dtype=np.uint8)
        idx = min(i, len(obj_mask_files) - 1)
        mask = self._read_and_resize_mask(obj_mask_files[idx])
        return mask

    def get_masks(self, i, use_init_mask=False):
        masks = []
        for obj_name in self.object_names:
            masks.append(
                self.get_mask(i, obj_name=obj_name, use_init_mask=use_init_mask)
            )
        return masks

    def get_init_masks(self):
        return self.get_masks(i=0, use_init_mask=True)

    def get_depth(self, i):
        depth = (
            cv2.imread(
                self.color_files[i].replace("rgb", "depth"), cv2.IMREAD_UNCHANGED
            )
            / 1e3
        )
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        # print(depth)
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(
            occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        if len(self.object_names) > 0:
            obj_name = self.object_names[0]
            ob_name = self.videoname_to_object.get(obj_name, obj_name)
        else:
            ob_name = self.videoname_to_object[self.get_video_name()]
        mesh = trimesh.load(
            f"/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/YCB_Video_Models/models/{ob_name}/textured_simple.obj"
        )
        return mesh


class YcbineoatReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None):
        self.video_dir = video_dir
        self.downscale = downscale
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))

        self.videoname_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
            "cracker_box": "003_cracker_box",
        }

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def __len__(self):
        return len(self.color_files)

    def get_gt_pose(self, i):
        try:
            pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
            return pose
        except:
            logging.info("GT pose not found, return None")
            return None

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        if color.shape[-1] == 4:
            color = color[..., :3]  # Drop alpha channel

        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_mask(self, i):
        mask = cv2.imread(self.color_files[i].replace("rgb", "gt_mask"), -1)
        if len(mask.shape) == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return mask

    def get_depth(self, i):
        depth = cv2.imread(self.color_files[i].replace("rgb", "depth"), -1) / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(
            occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        ob_name = self.videoname_to_object[self.get_video_name()]
        mesh = trimesh.load(
            f"/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/YCB_Video_Models/models/{ob_name}/textured_simple.obj"
        )
        return mesh


class Ho3dReader:
    def __init__(self, video_dir, ho3d_root) -> None:
        self.ho3d_root = ho3d_root
        self.video_dir = video_dir
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
        meta_file = self.color_files[0].replace(".jpg", ".pkl").replace("rgb", "meta")
        self.K = pickle.load(open(meta_file, "rb"))["camMat"]

        self.glcam_in_cvcam = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        self.id_strs = []
        for i in range(len(self.color_files)):
            id = os.path.basename(self.color_files[i]).split(".")[0]
            self.id_strs.append(id)

    def __len__(self):
        return len(self.color_files)

    def get_video_name(self):
        return os.path.dirname(os.path.abspath(self.color_files[0])).split("/")[-2]

    def get_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        mask = cv2.imread(f"{self.ho3d_root}/masks/{video_name}/{index:05d}.png", -1)
        return mask

    def get_occ_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        mask = cv2.imread(
            f"{self.ho3d_root}/masks/{video_name}_hand/{index:04d}.png", -1
        )
        return mask

    def get_gt_mesh(self):
        video2name = {
            "AP": "019_pitcher_base",
            "MPM": "010_potted_meat_can",
            "SB": "021_bleach_cleanser",
            "SM": "006_mustard_bottle",
        }
        video_name = self.get_video_name()
        for k in video2name:
            if video_name.startswith(k):
                ob_name = video2name[k]
                break
        mesh = trimesh.load(f"{self.ho3d_root}/models/{ob_name}/textured_simple.obj")
        return mesh

    def get_depth(self, i):
        depth_scale = 0.00012498664727900177  # meters per unit (~1/8000)
        path = self.color_files[i].replace(".jpg", ".png").replace("rgb", "depth")
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # If it's already 16UC1, just use it. Otherwise, recombine from BGR8.
        if depth.ndim == 2 and depth.dtype == np.uint16:
            depth16 = depth
        else:
            # depth is HxWx3, uint8; OpenCV = B,G,R
            # R = low byte, G = high byte
            low = depth[..., 2].astype(np.uint16)
            high = depth[..., 1].astype(np.uint16)
            depth16 = low | (high << 8)

        depth_m = depth16.astype(np.float32) * depth_scale
        return depth_m

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_gt_pose(self, i):
        meta_file = self.color_files[i].replace(".jpg", ".pkl").replace("rgb", "meta")
        meta = pickle.load(open(meta_file, "rb"))
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = self.glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt
