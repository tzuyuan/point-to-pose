"""
Exports live tracking output (RGB-D, mask, camera pose) to disk in a layout
``point2pose/model_tracking/reconstruct.py`` (ported from ``../kv_tracker``'s
``reconstruct.py``) reads directly for 2DGS/3DGS training -- no separate keyframe
pointcloud export needed; reconstruct.py backprojects whichever frames it needs
on the fly from all_frames_rgb/depth/mask/poses (see
``reconstruct.backproject_masked_depth`` / ``load_pairwise_check_frames``).

Unlike kv_tracker (which estimates pose with a monocular, scale-ambiguous method and
therefore needs a separate ``scale_align.py`` pass), point2pose backprojects RealSense
depth directly, so poses exported here are already metric -- no scale alignment step
is needed downstream.

Usage: owned and driven by the caller (e.g. the RealSense demo script), not by
``ModularPipeline`` itself, so it stays strictly opt-in.

    exporter = ReconstructionExporter(results_path)
    ...
    exporter.export_frame(frame, obj, frame.id)   # every tracked frame
"""

import os

import cv2
import numpy as np

from point2pose.utils.transform import inverse_SE3


class ReconstructionExporter:
    """Dumps per-frame RGB/depth/mask/pose under ``results_path``, mirroring
    kv_tracker's ``main.py --save_all_frames`` layout:

        results_path/
          intrinsics.npy            (3,3) camera intrinsics, saved once
          all_frames_idx.npy        (M,) original frame ids, appended every frame
          all_frames_poses.npy      (M,4,4) cam-to-world (T_wc), appended every frame
          all_frames_rgb/{id:06d}.png
          all_frames_depth/{id:06d}.png     (raw uint16, same units as the live Frame.depth)
          all_frames_mask/{id:06d}.png      (uint8, 0/255; only if frame.mask is available)
    """

    def __init__(self, results_path, save_all_frames=True, obj_id=0):
        self.results_path = str(results_path)
        self.save_all_frames = bool(save_all_frames)
        self.obj_id = int(obj_id)

        self.rgb_dir = os.path.join(self.results_path, "all_frames_rgb")
        self.depth_dir = os.path.join(self.results_path, "all_frames_depth")
        self.mask_dir = os.path.join(self.results_path, "all_frames_mask")
        os.makedirs(self.results_path, exist_ok=True)
        if self.save_all_frames:
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.mask_dir, exist_ok=True)

        self._intrinsics_saved = False
        self._all_frames_idx = []
        self._all_frames_poses = []

    # ------------------------------------------------------------------ #
    def export_frame(self, frame, obj, frame_id):
        """Export one tracked frame's RGB/depth/mask + the object's current camera pose.

        Args:
            frame: point2pose Frame (rgb HxWx3 uint8, depth HxW, intrinsics 3x3, mask
                optional [N,1,H,W] torch tensor).
            obj: the tracked Object; ``obj.pose`` is T_obj2cam (object/world -> camera).
            frame_id: int frame index, used for the export filenames.
        """
        if obj.pose is None:
            return

        if not self._intrinsics_saved and frame.intrinsics is not None:
            np.save(os.path.join(self.results_path, "intrinsics.npy"),
                    np.asarray(frame.intrinsics, dtype=np.float64))
            self._intrinsics_saved = True

        T_wc = inverse_SE3(np.asarray(obj.pose, dtype=np.float64))
        self._all_frames_poses.append(T_wc)
        self._all_frames_idx.append(int(frame_id))
        np.save(os.path.join(self.results_path, "all_frames_poses.npy"),
                np.stack(self._all_frames_poses, axis=0))
        np.save(os.path.join(self.results_path, "all_frames_idx.npy"),
                np.asarray(self._all_frames_idx, dtype=np.int64))

        if self.save_all_frames:
            fname = f"{int(frame_id):06d}.png"
            cv2.imwrite(os.path.join(self.rgb_dir, fname), frame.rgb[..., ::-1])
            if frame.depth is not None:
                cv2.imwrite(os.path.join(self.depth_dir, fname),
                            frame.depth.astype(np.uint16))
            mask_np = self._extract_mask_np(frame)
            if mask_np is not None:
                cv2.imwrite(os.path.join(self.mask_dir, fname),
                            ((mask_np > 0) * 255).astype(np.uint8))

    def _extract_mask_np(self, frame):
        """Returns frame.mask's raw values (SAM2 logits: >0 == inside mask, matching
        KeyFrameManager._extract_dense_pcd's ``mask > 0`` convention), not yet
        thresholded to bool -- callers must threshold with ``> 0`` themselves."""
        if frame.mask is None or frame.mask.shape[0] <= self.obj_id:
            return None
        m = frame.mask[self.obj_id, 0]
        return m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
