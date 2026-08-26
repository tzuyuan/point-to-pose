"""Shared loader for poster module visualizations (Point2Pose ECCV poster).

Supports the YCBMultiTrack-Real 3-object sequence and HO3D single-object
sequences. Loads saved run meta_data.npz + raw dataset frames.
"""
import os

import cv2
import numpy as np

FINAL = "/home/justin/results/eccv_point2pose/final_results"

def _ycb_real(seq, objs):
    return {
        "run": f"{FINAL}/ycbmultitrack_real_low_res_v2/{seq}",
        "data": f"/home/justin/data/YCBMultiTrack_recalib/{seq}",
        "obj_names": objs,
        "rgb": "{data}/rgb/{t:06d}.png",
        "depth": "{data}/depth/{t:06d}.png",
        "mask": "{data}/masks/{name}/{t:06d}.png",
        "K": "{data}/cam_K.txt",
        "wh": (640, 480),
    }


CONFIGS = {
    "ycb3b": _ycb_real(
        "021_bleach_cleanser_005_tomato_soup_can_008_pudding_box",
        ["021_bleach_cleanser", "005_tomato_soup_can", "008_pudding_box"]),
    "ycb2h": _ycb_real(
        "006_mustard_bottle_010_potted_meat_can_hard",
        ["006_mustard_bottle", "010_potted_meat_can"]),
    "ycb3": {
        "run": f"{FINAL}/ycbmultitrack_real_low_res_v2/006_mustard_bottle_010_potted_meat_can_005_tomato_soup_can",
        "data": "/home/justin/data/YCBMultiTrack_recalib/006_mustard_bottle_010_potted_meat_can_005_tomato_soup_can",
        "obj_names": ["006_mustard_bottle", "010_potted_meat_can", "005_tomato_soup_can"],
        "rgb": "{data}/rgb/{t:06d}.png",
        "depth": "{data}/depth/{t:06d}.png",
        "mask": "{data}/masks/{name}/{t:06d}.png",
        "K": "{data}/cam_K.txt",
        "wh": (640, 480),
    },
}


def _ho3d(seq):
    return {
        "run": f"{FINAL}/ho3d_all_final/{seq}",
        "data": f"/home/justin/data/HO3D_V3/evaluation/{seq}",
        "obj_names": [seq],
        "rgb": "{data}/rgb/{t:04d}.jpg",
        "depth": "{data}/depth/{t:04d}.png",
        "mask": "/home/justin/data/HO3D_V3/masks/{name}/{t:05d}.png",
        "K": None,
        "wh": (640, 480),
    }


for _s in ["AP10", "AP11", "AP12", "AP13", "AP14", "AP15",
           "MPM10", "MPM11", "MPM12", "MPM13", "MPM14",
           "SB11", "SB13", "SM1"]:
    CONFIGS[f"ho3d_{_s}"] = _ho3d(_s)

# Poster object colors (BGR for cv2): amber, magenta, spring green
OBJ_COLORS_BGR = [(0, 176, 255), (127, 38, 220), (98, 211, 60)]
OBJ_COLORS_RGB = [(c[2], c[1], c[0]) for c in OBJ_COLORS_BGR]

OUT_DIR = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"
os.makedirs(OUT_DIR, exist_ok=True)


class Run:
    def __init__(self, name="ycb3"):
        self.name = name
        cfg = CONFIGS[name]
        self.cfg = cfg
        self.run_dir = cfg["run"]
        self.data_dir = cfg["data"]
        self.obj_names = cfg["obj_names"]
        self.n_obj = len(self.obj_names)
        self.W, self.H = cfg["wh"]
        self.d = np.load(os.path.join(self.run_dir, "meta_data/meta_data.npz"),
                         allow_pickle=True)
        self.K = (np.loadtxt(cfg["K"].format(data=self.data_dir)).reshape(3, 3)
                  if cfg["K"] else None)
        self.n_frames = len(self.d["frame_id"])

    # --- raw data ---
    def rgb(self, t, rgb_order=False):
        im = cv2.imread(self.cfg["rgb"].format(data=self.data_dir, t=t))
        return im[:, :, ::-1].copy() if rgb_order else im

    def depth(self, t):
        return cv2.imread(self.cfg["depth"].format(data=self.data_dir, t=t),
                          cv2.IMREAD_UNCHANGED)

    def mask(self, t, obj):
        p = self.cfg["mask"].format(data=self.data_dir,
                                    name=self.obj_names[obj], t=t)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return (m > 127) if m is not None else None

    # --- run arrays ---
    def ragged(self, name, t):
        d = self.d
        off, ln = d[name + "_offsets"], d[name + "_lengths"]
        return d[name + "_data"][off[t]:off[t] + ln[t]]

    def track2d(self, t):
        return self.ragged("track2d", t).reshape(-1, 2)

    def track3d(self, t):
        return self.ragged("track3d", t).reshape(-1, 3)

    def visibles(self, t):
        return self.ragged("visibles", t).astype(bool)

    def valid(self, t):
        return np.asarray(self.d["valid"][t]).astype(bool)

    def keypoints_obj0(self, t):
        return self.ragged("obj_key_points", t).reshape(-1, 3)

    def kp_birth_frames_obj0(self, t):
        return self.ragged("obj_key_point_frames", t)

    def n_tracks(self, t):
        return len(self.track2d(t))

    def pose(self, t, obj=0):
        pa = self.d["obj_pose_all"]
        return pa[t, obj] if pa.ndim == 4 else self.d["obj_pose"][t]

    def track_obj_ids_voted(self, stride=10, cache=True):
        """Robust track->object assignment: majority vote of mask membership
        over all sampled frames where the track is visible."""
        cache_path = os.path.join(OUT_DIR, f"_ids_voted_{self.name}.npy")
        n_total = self.n_tracks(self.n_frames - 1)
        if cache and os.path.exists(cache_path):
            ids = np.load(cache_path)
            if len(ids) == n_total:
                return ids
        if self.n_obj == 1:
            ids = np.zeros(n_total, dtype=int)
            np.save(cache_path, ids)
            return ids
        votes = np.zeros((n_total, self.n_obj), dtype=int)
        for t in range(0, self.n_frames, stride):
            pts = self.track2d(t)
            v = self.visibles(t)
            xy = np.round(pts).astype(int)
            ok = (np.isfinite(pts).all(1) & v
                  & (xy[:, 0] >= 0) & (xy[:, 0] < self.W)
                  & (xy[:, 1] >= 0) & (xy[:, 1] < self.H))
            j = np.where(ok)[0]
            for o in range(self.n_obj):
                m = self.mask(t, o)
                if m is None:
                    continue
                dm = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                hit = dm[xy[j, 1], xy[j, 0]]
                votes[j[hit], o] += 1
        ids = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
        top, tot = votes.max(1), votes.sum(1)
        ids[(tot > 0) & (top / np.maximum(tot, 1) < 0.7)] = -1
        if cache:
            np.save(cache_path, ids)
        return ids

    def project(self, pts3d_cam):
        p = pts3d_cam @ self.K.T
        return p[:, :2] / p[:, 2:3]


def overlay_alpha(img, color_bgr, mask, alpha=0.45):
    out = img.copy()
    col = np.zeros_like(img)
    col[:] = color_bgr
    blend = cv2.addWeighted(img, 1 - alpha, col, alpha, 0)
    out[mask] = blend[mask]
    return out


def upscale(img, s=2):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
