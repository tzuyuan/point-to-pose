import os
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import trimesh

from point2pose.utils.transform import to_homo


def to_open3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def chamfer_distance_between_clouds_mutual(
    pred_pts: np.ndarray, gt_pts: np.ndarray
) -> np.ndarray:
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    pred_pts = pred_pts[np.isfinite(pred_pts).all(axis=1)]
    gt_pts = gt_pts[np.isfinite(gt_pts).all(axis=1)]
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.array([], dtype=np.float64)

    pcd_pred = to_open3d_cloud(pred_pts)
    pcd_gt = to_open3d_cloud(gt_pts)
    dist_pred_to_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    dist_gt_to_pred = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    return np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)


def _as_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            return None
        meshes = [g for g in mesh_or_scene.geometry.values()]
        return trimesh.util.concatenate(meshes)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    return None


def trimesh_clean(mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return None
    try:
        mesh.remove_infinite_values()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        return mesh
    except Exception:
        return None


def trimesh_split(mesh: trimesh.Trimesh, min_edge: int = 1000):
    if mesh is None:
        return []
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        return []
    return [c for c in comps if len(c.vertices) >= int(min_edge)]


def export_final_meshes_from_pipeline(
    pipeline, mesh_dir: str, prefix: str = "pred_mesh_obj"
) -> Dict[int, Optional[str]]:
    os.makedirs(mesh_dir, exist_ok=True)
    out: Dict[int, Optional[str]] = {}

    if not hasattr(pipeline, "objects"):
        return out

    sdf_builder = getattr(pipeline, "sdf_builder", None)
    for obj_idx, obj in enumerate(pipeline.objects):
        save_path = os.path.join(mesh_dir, f"{prefix}_{obj_idx}.ply")
        textured_save_path = os.path.join(mesh_dir, f"{prefix}_{obj_idx}_textured.glb")
        saved = False
        if sdf_builder is not None and hasattr(sdf_builder, "export_debug_mesh"):
            try:
                saved = bool(sdf_builder.export_debug_mesh(obj, save_path))
            except Exception:
                saved = False

        if not saved and getattr(obj, "sdf_volume", None) is not None:
            try:
                saved = bool(obj.sdf_volume.export_mesh(save_path))
            except Exception:
                saved = False

        # Also export a textured final mesh when possible.
        if sdf_builder is not None and hasattr(sdf_builder, "export_textured_mesh"):
            try:
                sdf_builder.export_textured_mesh(obj, textured_save_path)
            except Exception:
                pass

        out[obj_idx] = save_path if saved and os.path.exists(save_path) else None

    return out


def find_gt_visible_mesh_path(video_dir: str, obj_name: Optional[str] = None) -> Optional[str]:
    candidates = []
    if obj_name is not None:
        candidates.extend(
            [
                os.path.join(video_dir, "visible_mesh", f"{obj_name}.ply"),
                os.path.join(video_dir, f"visible_mesh_{obj_name}.ply"),
            ]
        )
    candidates.extend(
        [
            os.path.join(video_dir, "visible_mesh.ply"),
            os.path.join(video_dir, "visible_mesh", "visible_mesh.ply"),
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    visible_mesh_dir = os.path.join(video_dir, "visible_mesh")
    if os.path.isdir(visible_mesh_dir):
        mesh_files = sorted(
            [
                os.path.join(visible_mesh_dir, f)
                for f in os.listdir(visible_mesh_dir)
                if f.lower().endswith((".ply", ".obj", ".stl"))
            ]
        )
        if len(mesh_files) > 0:
            return mesh_files[0]
    return None


def _load_visible_geometry_points(
    mesh_path: str, voxel_size: float = 0.005, n_surface_samples: int = 120000
) -> np.ndarray:
    # Prefer mesh parsing, because many visible-mesh files are triangle meshes.
    tri_mesh = o3d.io.read_triangle_mesh(mesh_path)
    if tri_mesh is not None and len(tri_mesh.vertices) > 0:
        if len(tri_mesh.triangles) > 0:
            pcd = tri_mesh.sample_points_uniformly(number_of_points=n_surface_samples)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = tri_mesh.vertices
    else:
        pcd = o3d.io.read_point_cloud(mesh_path)

    if pcd is None or len(pcd.points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        if pcd is None or len(pcd.points) == 0:
            return np.empty((0, 3), dtype=np.float64)

    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


def evaluate_reconstructed_mesh(
    pred_mesh_path: Optional[str],
    gt_visible_mesh_path: Optional[str],
    output_dir: str,
    mesh_prefix: str,
    pred_pose_first: np.ndarray,
    gt_pose_first: np.ndarray,
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    if pred_mesh_path is None or gt_visible_mesh_path is None:
        return np.inf
    if not os.path.exists(pred_mesh_path) or not os.path.exists(gt_visible_mesh_path):
        return np.inf

    try:
        gt_pts = _load_visible_geometry_points(gt_visible_mesh_path, voxel_size=0.005)
        if len(gt_pts) == 0:
            return np.inf
        pcd = to_open3d_cloud(gt_pts)
        gt_down_path = os.path.join(output_dir, f"gt_{mesh_prefix}.ply")
        o3d.io.write_point_cloud(gt_down_path, pcd)
        gt_pts = np.asarray(pcd.points, dtype=np.float64).copy()

        pred_mesh = _as_mesh(trimesh.load(pred_mesh_path, force="mesh"))
        if pred_mesh is None:
            return np.inf

        pred_mesh.apply_transform(pred_pose_first)
        pred_mesh.apply_transform(np.linalg.inv(gt_pose_first))
        pred_tf_path = os.path.join(output_dir, f"pred_mesh_{mesh_prefix}.obj")
        pred_mesh.export(pred_tf_path)

        max_coord = gt_pts.max(axis=0).reshape(1, 3) + 0.3
        min_coord = gt_pts.min(axis=0).reshape(1, 3) - 0.3
        bad_mask = (pred_mesh.vertices > max_coord).any(axis=-1) | (
            pred_mesh.vertices < min_coord
        ).any(axis=-1)
        keep_mask = ~bad_mask
        if int(keep_mask.sum()) < 3:
            return np.inf
        pred_mesh.update_vertices(keep_mask)
        pred_mesh = trimesh_clean(pred_mesh)
        if pred_mesh is None:
            return np.inf

        pred_clean_path = os.path.join(output_dir, f"pred_mesh_cleaned_{mesh_prefix}.obj")
        pred_mesh.export(pred_clean_path)

        components = trimesh_split(pred_mesh, min_edge=1000)
        if len(components) == 0:
            components = trimesh_split(pred_mesh, min_edge=3)
        if len(components) == 0:
            return np.inf

        best_component = None
        best_size = 0
        for component in components:
            dists = np.linalg.norm(component.vertices, axis=-1)
            if dists.min() > 0.1:
                continue
            if len(component.vertices) > best_size:
                best_size = len(component.vertices)
                best_component = component
        if best_component is None:
            best_component = max(components, key=lambda c: len(c.vertices))
        pred_mesh = best_component

        pred_biggest_path = os.path.join(output_dir, f"pred_mesh_biggest_{mesh_prefix}.obj")
        pred_mesh.export(pred_biggest_path)

        pred_pts, _ = trimesh.sample.sample_surface(
            pred_mesh, 99999, face_weight=None, sample_color=False
        )
        pred_pts = np.asarray(pred_pts, dtype=np.float64)
        pred_pts = pred_pts[np.isfinite(pred_pts).all(axis=1)]
        if len(pred_pts) == 0:
            return np.inf

        pcd_pred = to_open3d_cloud(pred_pts).voxel_down_sample(0.005)
        if pcd_pred is None or len(pcd_pred.points) == 0:
            return np.inf
        pred_pts = np.asarray(pcd_pred.points, dtype=np.float64)
        pcd_gt = to_open3d_cloud(gt_pts)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_pred,
            pcd_gt,
            0.02,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pred_pts_icp = (reg_p2p.transformation @ to_homo(pred_pts).T).T[:, :3]
        chamfer_dists = chamfer_distance_between_clouds_mutual(pred_pts_icp, gt_pts)
        if len(chamfer_dists) == 0:
            return np.inf
        cd = float(chamfer_dists.mean() * 100.0)
        if not np.isfinite(cd):
            return np.inf
        return cd
    except Exception:
        return np.inf
