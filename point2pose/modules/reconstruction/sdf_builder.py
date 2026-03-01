import numpy as np
import os

from numba import njit, prange
import open3d as o3d
import trimesh
from skimage import measure
from typing import Optional

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    from pycuda.compiler import SourceModule

    FUSION_GPU_MODE = 1
except Exception as err:
    print(f"Warning: {err}")
    print("Failed to import PyCUDA. Running fusion in CPU mode.")
    FUSION_GPU_MODE = 0


def rigid_transform(xyz, transform):
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


class TSDFVolume:
    """Volumetric TSDF fusion of RGB-D images."""

    def __init__(self, vol_bnds, voxel_size, use_gpu=True):
        vol_bnds = np.asarray(vol_bnds, dtype=np.float32)
        assert vol_bnds.shape == (3, 2), "`vol_bnds` should be (3, 2)"

        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size
        self._color_const = 256 * 256

        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = bool(use_gpu and FUSION_GPU_MODE)

        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            self._cuda_src_mod = SourceModule(
                """
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }"""
            )
            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(
                np.ceil(
                    float(np.prod(self._vol_dim))
                    / float(self._max_gpu_threads_per_block)
                )
            )
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(
                gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x)))
            )
            grid_dim_z = min(
                gpu_dev.MAX_GRID_DIM_Z,
                int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))),
            )
            self._max_gpu_grid_dim = np.array(
                [grid_dim_x, grid_dim_y, grid_dim_z]
            ).astype(int)
            self._n_gpu_loops = int(
                np.ceil(
                    float(np.prod(self._vol_dim))
                    / float(
                        np.prod(self._max_gpu_grid_dim)
                        * self._max_gpu_threads_per_block
                    )
                )
            )
        else:
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing="ij",
            )
            self.vox_coords = (
                np.concatenate(
                    [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
                )
                .astype(int)
                .T
            )

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(
        self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.0, obj_mask=None
    ):
        del obj_mask
        im_h, im_w = depth_im.shape
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        if self.gpu_mode:
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(
                    self._tsdf_vol_gpu,
                    self._weight_vol_gpu,
                    self._color_vol_gpu,
                    cuda.InOut(self._vol_dim.astype(np.float32)),
                    cuda.InOut(self._vol_origin.astype(np.float32)),
                    cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                    cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                    cuda.InOut(
                        np.asarray(
                            [
                                gpu_loop_idx,
                                self._voxel_size,
                                im_h,
                                im_w,
                                self._trunc_margin,
                                obs_weight,
                            ],
                            np.float32,
                        )
                    ),
                    cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                    cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                    block=(self._max_gpu_threads_per_block, 1, 1),
                    grid=(
                        int(self._max_gpu_grid_dim[0]),
                        int(self._max_gpu_grid_dim[1]),
                        int(self._max_gpu_grid_dim[2]),
                    ),
                )
            return

        cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
        cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = self.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape, dtype=np.float32)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.minimum(1, depth_diff / self._trunc_margin)
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]
        tsdf_vol_new, w_new = self.integrate_tsdf(
            tsdf_vals, valid_dist.astype(np.float32), w_old, obs_weight
        )
        self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = np.floor(old_color / self._color_const)
        old_g = np.floor((old_color - old_b * self._color_const) / 256)
        old_r = old_color - old_b * self._color_const - old_g * 256
        new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
        new_b = np.floor(new_color / self._color_const)
        new_g = np.floor((new_color - new_b * self._color_const) / 256)
        new_r = new_color - new_b * self._color_const - new_g * 256
        new_b = np.minimum(
            255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
        )
        new_g = np.minimum(
            255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
        )
        new_r = np.minimum(
            255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
        )
        self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
            new_b * self._color_const + new_g * 256 + new_r
        )

    def get_volume(self):
        if self.gpu_mode:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu


class NvbloxVolume:
    """
    Thin adapter that keeps the same interface used by SDFBuilder.
    Falls back to legacy TSDF when nvblox is unavailable or API mismatches.
    """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True):
        self._vol_bnds = np.asarray(vol_bnds, dtype=np.float32)
        self._voxel_size = float(voxel_size)
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)
        self._color_const = 256 * 256
        self._use_gpu = bool(use_gpu)
        self._mapper = self._build_mapper()

    def _build_mapper(self):
        try:
            import nvblox_torch  # type: ignore
        except Exception as exc:
            raise ImportError(f"nvblox_torch import failed: {exc}") from exc

        # nvblox Python APIs differ across builds; try common constructors.
        ctor = getattr(nvblox_torch, "Mapper", None)
        if ctor is None:
            raise RuntimeError("nvblox_torch.Mapper not found in installed package")

        pmin = self._vol_bnds[:, 0]
        pmax = self._vol_bnds[:, 1]
        init_variants = [
            {"voxel_size": self._voxel_size, "aabb_min": pmin, "aabb_max": pmax},
            {"voxel_size": self._voxel_size, "min_corner": pmin, "max_corner": pmax},
            {
                "voxel_size": self._voxel_size,
                "device": "cuda" if self._use_gpu else "cpu",
            },
            {},
        ]
        last_exc = None
        for kwargs in init_variants:
            try:
                return ctor(**kwargs)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"Failed to construct nvblox mapper: {last_exc}"
        ) from last_exc

    def _try_call(self, obj, names, kwargs):
        for name in names:
            fn = getattr(obj, name, None)
            if fn is None:
                continue
            try:
                return fn(**kwargs)
            except TypeError:
                continue
        return None

    def integrate(
        self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.0, obj_mask=None
    ):
        del obs_weight
        depth_use = np.asarray(depth_im, dtype=np.float32)
        if obj_mask is not None:
            m = np.asarray(obj_mask)
            depth_use = depth_use.copy()
            depth_use[m <= 0] = 0.0

        color_use = np.asarray(color_im)
        K = np.asarray(cam_intr, dtype=np.float32)
        T = np.asarray(cam_pose, dtype=np.float32)

        out = self._try_call(
            self._mapper,
            ["integrate_depth", "integrate_frame", "integrate"],
            {
                "depth_image": depth_use,
                "color_image": color_use,
                "T_L_C": T,
                "camera_matrix": K,
            },
        )
        if out is None:
            # try positional compatibility for some wrappers
            for name in ["integrate_depth", "integrate_frame", "integrate"]:
                fn = getattr(self._mapper, name, None)
                if fn is None:
                    continue
                try:
                    fn(depth_use, color_use, K, T)
                    return
                except Exception:
                    continue
            raise RuntimeError("No compatible nvblox integration API found")

    def get_volume(self):
        # nvblox keeps sparse layers internally; dense export is optional.
        return None, None

    def query_sdf(self, pts_obj: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(pts_obj, dtype=np.float32)
        for name in ["query_sdf", "query_tsdf", "interpolate_tsdf"]:
            fn = getattr(self._mapper, name, None)
            if fn is None:
                continue
            try:
                vals = fn(pts)
            except Exception:
                continue
            vals = np.asarray(vals).reshape(-1)
            if vals.shape[0] == pts.shape[0]:
                return vals.astype(np.float32)
        return None

    def export_mesh(self, save_path):
        for name in ["save_mesh", "save_triangle_mesh", "export_mesh"]:
            fn = getattr(self._mapper, name, None)
            if fn is None:
                continue
            try:
                fn(save_path)
                return True
            except Exception:
                continue
        return False


class SDFBuilder:
    """
    Incremental SDF builder for each tracked object.
    Uses keyframe dense observations and stores the latest TSDF arrays on object.sdf.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("build_sdf_after_global_opt", True))
        self.fuse_color = bool(cfg.get("sdf_fuse_color", True))
        self.use_gpu = bool(cfg.get("sdf_use_gpu", True))
        self.backend = str(cfg.get("sdf_backend", "nvblox")).lower()
        self.voxel_size = float(cfg.get("sdf_voxel_size", 0.005))
        self.padding = float(cfg.get("sdf_bounds_padding", 0.02))
        self.min_points = int(cfg.get("sdf_min_points", 300))
        self.max_radius = float(cfg.get("sdf_filter_max_radius", 0.25))
        self.keep_percentile = float(cfg.get("sdf_filter_keep_percentile", 98.0))

    def _filter_points(self, pts_obj):
        if pts_obj.shape[0] == 0:
            return pts_obj
        center = np.median(pts_obj, axis=0)
        dist = np.linalg.norm(pts_obj - center[None, :], axis=1)
        if dist.size == 0:
            return pts_obj
        dist_th = np.percentile(dist, self.keep_percentile)
        dist_th = min(dist_th, self.max_radius)
        keep = dist <= dist_th
        return pts_obj[keep]

    def _init_object_volume(self, obj, pts_obj):
        pmin = pts_obj.min(axis=0) - self.padding
        pmax = pts_obj.max(axis=0) + self.padding
        vol_bnds = np.stack([pmin, pmax], axis=1).astype(np.float32)
        if self.backend == "nvblox":
            try:
                obj.sdf_volume = NvbloxVolume(
                    vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=self.use_gpu
                )
            except Exception as exc:
                print(
                    f"Warning: failed to initialize nvblox backend ({exc}), fallback to TSDFVolume."
                )
                self.backend = "legacy"
                obj.sdf_volume = TSDFVolume(
                    vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=self.use_gpu
                )
        else:
            obj.sdf_volume = TSDFVolume(
                vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=self.use_gpu
            )
        obj.sdf_num_integrated = 0

        print(
            f"Initialized SDF volume for object {getattr(obj, 'obj_id', 'backend is: '+self.backend)} with bounds {vol_bnds} and voxel size {self.voxel_size}"
        )

    def integrate_keyframe(self, obj, keyframe):
        if not self.enabled:
            return False
        if (
            keyframe is None
            or keyframe.dense_pts is None
            or keyframe.dense_pts.shape[0] == 0
        ):
            return False

        T_c2o = np.linalg.inv(keyframe.pose)
        pts_obj = rigid_transform(
            np.asarray(keyframe.dense_pts, dtype=np.float32), T_c2o.astype(np.float32)
        )
        pts_obj = self._filter_points(pts_obj)
        if pts_obj.shape[0] < self.min_points:
            return False

        if getattr(obj, "sdf_volume", None) is None:
            self._init_object_volume(obj, pts_obj)

        frame = keyframe.frame
        if frame is None or frame.depth is None:
            return False

        depth_m = frame.depth.astype(np.float32) / float(frame.depth_factor)
        if (
            getattr(frame, "mask", None) is not None
            and frame.mask.shape[0] > keyframe.obj_id
        ):
            mask = frame.mask[keyframe.obj_id, 0]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            elif hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            else:
                mask = np.asarray(mask)
            depth_m = depth_m.copy()
            depth_m[mask <= 0] = 0.0
        else:
            mask = None

        color_im = None
        if self.fuse_color and getattr(frame, "rgb", None) is not None:
            color_im = np.asarray(frame.rgb)
        else:
            # Keep geometry integration working even when RGB is unavailable or disabled.
            h, w = depth_m.shape
            color_im = np.zeros((h, w, 3), dtype=np.uint8)

        cam_pose_obj = np.linalg.inv(np.asarray(keyframe.pose, dtype=np.float32))
        obj.sdf_volume.integrate(
            color_im=color_im,
            depth_im=depth_m,
            cam_intr=np.asarray(frame.intrinsics, dtype=np.float32),
            cam_pose=cam_pose_obj,
            obs_weight=1.0,
            obj_mask=mask,
        )
        obj.sdf_num_integrated = int(getattr(obj, "sdf_num_integrated", 0)) + 1

        tsdf_vol, color_vol = obj.sdf_volume.get_volume()
        obj.sdf = {
            "backend": self.backend,
            "fuse_color": bool(self.fuse_color),
            "vol_bnds": obj.sdf_volume._vol_bnds.copy(),
            "vol_origin": obj.sdf_volume._vol_origin.copy(),
            "voxel_size": float(obj.sdf_volume._voxel_size),
            "num_integrated": int(obj.sdf_num_integrated),
        }
        if tsdf_vol is not None:
            obj.sdf["tsdf"] = tsdf_vol.copy()
        if color_vol is not None:
            obj.sdf["color"] = color_vol.copy()
        return True

    def export_debug_mesh(self, obj, save_path):
        if getattr(obj, "sdf_volume", None) is None:
            return False
        try:
            if hasattr(obj.sdf_volume, "export_mesh") and obj.sdf_volume.export_mesh(
                save_path
            ):
                return True

            mesh = self._build_colored_o3d_mesh(obj)
            if mesh is None:
                return False
            o3d.io.write_triangle_mesh(save_path, mesh, write_ascii=False)
            return True
        except Exception:
            return False

    def _build_colored_o3d_mesh(self, obj):
        tsdf_vol, color_vol = obj.sdf_volume.get_volume()
        if tsdf_vol is None:
            return None

        verts, faces, norms, _ = measure.marching_cubes(
            tsdf_vol, level=0, method="lewiner"
        )
        verts_xyz = verts * obj.sdf_volume._voxel_size + obj.sdf_volume._vol_origin

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_xyz.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms.astype(np.float64))

        if color_vol is not None:
            verts_ind = np.clip(
                np.round(verts).astype(int), 0, np.array(tsdf_vol.shape) - 1
            )
            rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            cconst = obj.sdf_volume._color_const
            colors_b = np.floor(rgb_vals / cconst)
            colors_g = np.floor((rgb_vals - colors_b * cconst) / 256)
            colors_r = rgb_vals - colors_b * cconst - colors_g * 256
            colors = (
                np.stack([colors_r, colors_g, colors_b], axis=1).astype(np.float32)
                / 255.0
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return mesh

    def export_textured_mesh(self, obj, save_path):
        if getattr(obj, "sdf_volume", None) is None:
            return False
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Let backend-specific exporters handle the target path if available.
            if hasattr(obj.sdf_volume, "export_mesh") and obj.sdf_volume.export_mesh(
                save_path
            ):
                return True

            mesh = self._build_colored_o3d_mesh(obj)
            if mesh is None:
                return False

            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            if verts.size == 0 or faces.size == 0:
                return False

            kwargs = {"process": False}
            if len(mesh.vertex_normals) == len(mesh.vertices):
                kwargs["vertex_normals"] = np.asarray(mesh.vertex_normals)
            if len(mesh.vertex_colors) == len(mesh.vertices):
                colors = np.asarray(mesh.vertex_colors)
                colors = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
                if colors.shape[1] == 3:
                    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
                    colors = np.hstack([colors, alpha])
                kwargs["vertex_colors"] = colors

            tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, **kwargs)
            tri_mesh.export(save_path)
            return os.path.exists(save_path)
        except Exception:
            return False
