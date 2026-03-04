import sys
from pathlib import Path
import time
import os
import argparse
import shutil

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
from omegaconf import OmegaConf
import torch
import numpy as np
import trimesh

from point2pose.io.sources.dataset.datareader import YCBInIsaacReader
from point2pose.pipeline.pipeline_single_process import PipelineSingleProcess
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.data_types.frame import Frame
from point2pose.utils.transform import inverse_SE3
from point2pose.utils.visualization import (
    visualize_and_save_tracking_results,
    visualize_and_save_tracking_results_with_gt,
)
from point2pose.utils.evaluation import (
    adi_err,
    add_err,
    compute_auc,
    plot_evaluation_results,
    plot_error_over_time,
    plot_pose_errors,
    plot_pose_error_comparison,
    plot_recall_vs_threshold,
)
from point2pose.utils.mesh_eval import (
    export_final_meshes_from_pipeline,
    evaluate_reconstructed_mesh,
    find_gt_visible_mesh_path,
)


def _canonical_object_name(reader, obj_name):
    # Try explicit map first, then use the provided directory name.
    return reader.videoname_to_object.get(obj_name, obj_name)


def load_ycb_mesh(reader, model_root, obj_name):
    """
    Load a YCB object mesh for one object name.
    """
    ob_name = _canonical_object_name(reader, obj_name)

    # Common YCB model structure: {model_root}/{ob_name}/textured_simple.obj
    # or {model_root}/models/{ob_name}/textured_simple.obj

    candidates = [
        os.path.join(model_root, ob_name, "textured.obj"),
        os.path.join(model_root, "models", ob_name, "textured_simple.obj"),
        os.path.join(
            model_root, ob_name, "google_16k", "textured.obj"
        ),  # Some versions
    ]

    mesh_path = None
    for p in candidates:
        if os.path.exists(p):
            mesh_path = p
            break

    if mesh_path is None:
        raise FileNotFoundError(
            f"Could not find mesh for {ob_name} in {model_root}. Checked: {candidates}"
        )

    print(f"Loading mesh for {obj_name} from {mesh_path}")
    return trimesh.load(mesh_path)


def gt_bbox_from_mesh(mesh, mode="aabb"):
    """
    Build a bbox representation for visualization.
    mode:
      - 'aabb'    : mesh-axis aligned min/max in mesh frame (legacy behavior)
      - 'obb_fit' : fitted oriented bbox (PCA) from mesh geometry
    """
    mode = str(mode).strip().lower()
    if mode in ("obb_fit", "obb", "oriented", "fit_oriented"):
        try:
            verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
            if verts.shape[0] < 3:
                raise ValueError("not enough vertices to fit OBB")

            # PCA frame in mesh coordinates.
            mean = verts.mean(axis=0)
            centered = verts - mean.reshape(1, 3)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            R = np.asarray(eigvecs[:, order], dtype=float).reshape(3, 3)
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1.0

            # Tight bounds in PCA frame.
            verts_local = (R.T @ centered.T).T
            mn_local = verts_local.min(axis=0)
            mx_local = verts_local.max(axis=0)
            extent = (mx_local - mn_local).astype(float)
            center_local = 0.5 * (mn_local + mx_local)
            center = (mean + R @ center_local).astype(float)

            return {
                "center": center,
                "extent": extent,
                "rot": R.astype(float),
            }
        except Exception as e:
            print(
                f"[Visualization] Failed to fit oriented bbox from mesh ({e}), "
                "falling back to axis-aligned mesh bbox."
            )

    bmin, bmax = mesh.bounds  # shape (3,), (3,)
    return np.vstack([bmin.astype(float), bmax.astype(float)])  # (2,3)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _parse_bgr_color(value, default=(0, 255, 255)):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError("color must have 3 components")
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return tuple(int(x) for x in arr.tolist())
    except Exception:
        return tuple(int(x) for x in default)


def load_is_obj_in_image_labels(reader, video_path, obj_name, num_frames):
    """
    Load per-frame object visibility labels for evaluation filtering.
    """
    labels = _load_binary_labels(
        reader=reader,
        video_path=video_path,
        obj_name=obj_name,
        label_root="is_obj_in_image_labels",
        label_file_name="is_obj_in_image.npy",
    )
    if labels.shape[0] != num_frames:
        print(
            f"[{obj_name}] Visibility label length mismatch: "
            f"{labels.shape[0]} vs num_frames={num_frames}. Adjusting to frame count."
        )
        if labels.shape[0] < num_frames:
            padded = np.zeros((num_frames,), dtype=bool)
            padded[: labels.shape[0]] = labels
            labels = padded
        else:
            labels = labels[:num_frames]
    return labels


def _load_binary_labels(
    reader,
    video_path,
    obj_name,
    label_root,
    label_file_name,
):
    canonical_name = _canonical_object_name(reader, obj_name)
    candidates = [
        os.path.join(video_path, label_root, obj_name, label_file_name),
        os.path.join(video_path, label_root, canonical_name, label_file_name),
    ]
    label_path = None
    for path in candidates:
        if os.path.exists(path):
            label_path = path
            break
    if label_path is None:
        raise FileNotFoundError(
            f"Could not find {label_root}/{label_file_name} for object {obj_name}. "
            f"Checked: {candidates}"
        )
    return np.asarray(np.load(label_path)).reshape(-1) > 0


def load_is_mask_visible_labels(reader, video_path, obj_name, num_frames):
    """
    Load per-frame mask-visibility labels for evaluation filtering.
    """
    labels = _load_binary_labels(
        reader=reader,
        video_path=video_path,
        obj_name=obj_name,
        label_root="is_mask_visible",
        label_file_name="is_mask_visible.npy",
    )
    if labels.shape[0] != num_frames:
        print(
            f"[{obj_name}] Mask-visibility label length mismatch: "
            f"{labels.shape[0]} vs num_frames={num_frames}. Adjusting to frame count."
        )
        if labels.shape[0] < num_frames:
            padded = np.zeros((num_frames,), dtype=bool)
            padded[: labels.shape[0]] = labels
            labels = padded
        else:
            labels = labels[:num_frames]
    return labels


def run_ycbineoat_single(
    data_path: str, video_name: str, out_dir: str, config_path: str, model_path: str
):
    # data_path should be the root containing the video folders
    # video_name is the folder name (e.g. bleach0)
    video_path = os.path.join(data_path, video_name)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    reader = YCBInIsaacReader(video_path)

    # We use the video name from the reader which is derived from the path
    video_name = reader.get_video_name()

    out_folder = os.path.join(out_dir, video_name, "")
    vis_folder = os.path.join(out_folder, "output_images")
    with_gt_folder = os.path.join(out_folder, "with_gt")
    mesh_folder = os.path.join(out_folder, "mesh")
    vis_folder_reg_corr = os.path.join(out_folder, "registration_correspondence")
    with_gt_folder_reg_corr = os.path.join(
        out_folder, "with_gt_registration_correspondence"
    )

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(with_gt_folder, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(vis_folder_reg_corr, exist_ok=True)
    os.makedirs(with_gt_folder_reg_corr, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    vis_cfg = cfg.visualization.params
    bbox_frame = str(vis_cfg.get("bbox_frame", "mesh")).strip().lower()
    if bbox_frame not in ("mesh", "center"):
        print(
            f"[Visualization] Unsupported bbox_frame='{bbox_frame}', defaulting to 'mesh'. "
            "Supported: ['mesh', 'center']."
        )
        bbox_frame = "mesh"

    bbox_fit_mode = str(vis_cfg.get("bbox_fit_mode", "aabb")).strip().lower()
    if bbox_fit_mode in ("obb", "oriented", "fit_oriented"):
        bbox_fit_mode = "obb_fit"
    if bbox_fit_mode not in ("aabb", "obb_fit"):
        print(
            f"[Visualization] Unsupported bbox_fit_mode='{bbox_fit_mode}', defaulting to 'aabb'. "
            "Supported: ['aabb', 'obb_fit']."
        )
        bbox_fit_mode = "aabb"

    project_mesh_contour = _as_bool(
        vis_cfg.get(
            "project_mesh_contour",
            vis_cfg.get("visualize_mesh_contour", False),
        ),
        default=False,
    )
    mesh_contour_linewidth = int(vis_cfg.get("mesh_contour_linewidth", 2))
    if mesh_contour_linewidth < 1:
        mesh_contour_linewidth = 1
    mesh_contour_line_color = _parse_bgr_color(
        vis_cfg.get("mesh_contour_line_color", [0, 255, 255]),
        default=(0, 255, 255),
    )

    cfg.pipeline.params.sdf_mesh_save_dir = mesh_folder
    cfg.pipeline.params.sdf_mesh_save_every = 1
    object_names = reader.get_object_names()
    print(f"Found objects in video {video_name}: {object_names}")
    if len(object_names) == 0:
        raise RuntimeError(
            f"No objects found under {os.path.join(video_path, 'masks')}"
        )

    cfg.pipeline.params.max_num_obj = len(object_names)
    # We provide dataset masks directly.
    cfg.pipeline.params.use_segmenter = False

    # Set meta_data_save_path to save in the results folder for this dataset
    if cfg.pipeline.params.get("save_meta_data", True):
        meta_data_folder = os.path.join(out_folder, "meta_data")
        cfg.pipeline.params.meta_data_save_path = meta_data_folder

    if cfg.pipeline.type == "single_process":
        pipeline = PipelineSingleProcess(cfg)
    elif cfg.pipeline.type == "modular":
        pipeline = ModularPipeline(cfg)
    else:
        raise ValueError(f"Invalid pipeline type: {cfg.pipeline.type}")

    # Load mesh and bbox for each object.
    meshes = {}
    gt_bbox_minmax_by_object = {}
    mesh_vertices_by_index = {}
    mesh_faces_by_index = {}
    is_obj_in_image_labels_by_object = {}
    is_mask_visible_labels_by_object = {}
    eval_labels_by_object = {}
    for obj_idx, obj_name in enumerate(object_names):
        mesh = load_ycb_mesh(reader, model_path, obj_name)
        meshes[obj_name] = mesh
        gt_bbox_minmax_by_object[obj_name] = gt_bbox_from_mesh(mesh, mode=bbox_fit_mode)
        mesh_vertices_by_index[obj_idx] = np.asarray(mesh.vertices, dtype=np.float64)
        mesh_faces_by_index[obj_idx] = np.asarray(mesh.faces, dtype=np.int64)
        missing_label_files = []
        try:
            is_obj_in_image_labels_by_object[obj_name] = load_is_obj_in_image_labels(
                reader=reader,
                video_path=video_path,
                obj_name=obj_name,
                num_frames=len(reader),
            )
        except FileNotFoundError:
            is_obj_in_image_labels_by_object[obj_name] = None
            missing_label_files.append("is_obj_in_image_labels/is_obj_in_image.npy")

        try:
            is_mask_visible_labels_by_object[obj_name] = load_is_mask_visible_labels(
                reader=reader,
                video_path=video_path,
                obj_name=obj_name,
                num_frames=len(reader),
            )
        except FileNotFoundError:
            is_mask_visible_labels_by_object[obj_name] = None
            missing_label_files.append("is_mask_visible/is_mask_visible.npy")

        eval_mask = np.ones((len(reader),), dtype=bool)
        if is_obj_in_image_labels_by_object[obj_name] is not None:
            eval_mask &= is_obj_in_image_labels_by_object[obj_name]
        if is_mask_visible_labels_by_object[obj_name] is not None:
            eval_mask &= is_mask_visible_labels_by_object[obj_name]
        if len(missing_label_files) > 0:
            print(
                f"[{video_name}/{obj_name}] Missing visibility labels: "
                f"{', '.join(missing_label_files)}. "
                "Evaluating all GT poses for missing criteria."
            )
        eval_labels_by_object[obj_name] = eval_mask

    out_poses_by_object = {obj_name: [] for obj_name in object_names}
    gt_poses_by_object = {obj_name: [] for obj_name in object_names}
    eval_ids_by_object = {obj_name: [] for obj_name in object_names}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(reader)):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)

        # Use first mask in each object folder for initialization frame.
        masks = reader.get_masks(i, use_init_mask=(i == 0))
        mask = np.stack(masks, axis=0)  # [N,H,W]
        mask = torch.from_numpy(mask).unsqueeze(1).to(device)  # [N,1,H,W]

        frame = Frame(
            id=i,
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=reader.K,
            depth_factor=1.0,  # Reader already converts to meters
            timestamp=time.time(),
        )

        out_pose = pipeline.step(frame)
        if out_pose.ndim == 2:
            out_pose = out_pose.reshape(1, 4, 4)
        n_from_pipeline = min(len(object_names), out_pose.shape[0])

        gt_pose_map = reader.get_gt_poses(i)

        for obj_idx, obj_name in enumerate(object_names[:n_from_pipeline]):
            pred_pose = out_pose[obj_idx].reshape(4, 4)
            out_poses_by_object[obj_name].append(pred_pose)
            gt_pose = gt_pose_map.get(obj_name, None)
            gt_poses_by_object[obj_name].append(gt_pose)
            if eval_labels_by_object[obj_name][i]:
                eval_ids_by_object[obj_name].append(i)
            if i == 0 and gt_pose is not None and obj_idx < len(pipeline.objects):
                pipeline.objects[obj_idx].init_pose = gt_pose

        gt_poses = []
        gt_indices = []
        gt_bbox_by_index = {}
        for obj_idx, obj_name in enumerate(object_names[:n_from_pipeline]):
            gt_pose = gt_pose_map.get(obj_name, None)
            if gt_pose is None:
                continue
            gt_indices.append(obj_idx)
            gt_poses.append(gt_pose)
            gt_bbox_by_index[obj_idx] = gt_bbox_minmax_by_object[obj_name]

        display_frame = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method=vis_cfg.points_vis_method,
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_by_index,
            mesh_vertices_by_object=mesh_vertices_by_index,
            mesh_faces_by_object=mesh_faces_by_index,
            project_mesh_contour=project_mesh_contour,
            mesh_contour_line_color=mesh_contour_line_color,
            mesh_contour_linewidth=mesh_contour_linewidth,
            bbox_frame=bbox_frame,
        )

        gt_overlay_frame = visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame,
            gt_poses=gt_poses,
            gt_object_indices=gt_indices,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method=vis_cfg.points_vis_method,
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_by_index,
            mesh_vertices_by_object=mesh_vertices_by_index,
            mesh_faces_by_object=mesh_faces_by_index,
            project_mesh_contour=project_mesh_contour,
            mesh_contour_line_color=mesh_contour_line_color,
            mesh_contour_linewidth=mesh_contour_linewidth,
            gt_bbox_min_max_by_object=gt_bbox_by_index,
            pred_pose_color=(0, 255, 0),
            bbox_frame=bbox_frame,
        )

        if vis_cfg.save_images:
            out_path = os.path.join(with_gt_folder, f"frame_{i:06d}.png")
            cv2.imwrite(out_path, gt_overlay_frame)

        display_frame_reg_corr = visualize_and_save_tracking_results(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="registration_correspondence",
            save_images=vis_cfg.save_images,
            output_image_dir=vis_folder_reg_corr,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_by_index,
            mesh_vertices_by_object=mesh_vertices_by_index,
            mesh_faces_by_object=mesh_faces_by_index,
            project_mesh_contour=project_mesh_contour,
            mesh_contour_line_color=mesh_contour_line_color,
            mesh_contour_linewidth=mesh_contour_linewidth,
            bbox_frame=bbox_frame,
        )

        gt_overlay_frame_reg_corr = visualize_and_save_tracking_results_with_gt(
            frame=frame,
            objects=pipeline.objects,
            track_table=pipeline.track_table,
            est_result_frame=display_frame_reg_corr,
            gt_poses=gt_poses,
            gt_object_indices=gt_indices,
            frame_id=i,
            visualize_points=vis_cfg.visualize_points,
            points_vis_method="registration_correspondence",
            save_images=vis_cfg.save_images,
            output_image_dir=with_gt_folder_reg_corr,
            camera_intrinsics=reader.K,
            bbox_min_max=gt_bbox_by_index,
            mesh_vertices_by_object=mesh_vertices_by_index,
            mesh_faces_by_object=mesh_faces_by_index,
            project_mesh_contour=project_mesh_contour,
            mesh_contour_line_color=mesh_contour_line_color,
            mesh_contour_linewidth=mesh_contour_linewidth,
            gt_bbox_min_max_by_object=gt_bbox_by_index,
            pred_pose_color=(0, 255, 0),
            bbox_frame=bbox_frame,
        )

        if vis_cfg.save_images:
            out_path = os.path.join(with_gt_folder_reg_corr, f"frame_{i:06d}.png")
            cv2.imwrite(out_path, gt_overlay_frame_reg_corr)

    per_object_results = {}
    all_adi_errs = []
    all_add_errs = []
    total_eval_frames = 0
    for obj_name in object_names:
        eval_ids = eval_ids_by_object[obj_name]
        if len(eval_ids) == 0:
            print(
                f"[{obj_name}] No frames after applying available visibility filtering, skipping object evaluation."
            )
            continue

        pred_poses = []
        gt_poses = []
        for frame_id in eval_ids:
            gt_pose = gt_poses_by_object[obj_name][frame_id]
            if gt_pose is None:
                continue
            pred_poses.append(out_poses_by_object[obj_name][frame_id])
            gt_poses.append(gt_pose)
        pred_poses = np.array(pred_poses)
        gt_poses = np.array(gt_poses)
        if len(pred_poses) == 0 or len(gt_poses) == 0:
            print(
                f"[{obj_name}] Not enough valid poses in visible frames for evaluation."
            )
            continue

        # Align first valid frame for this object.
        pred_poses_raw = pred_poses.copy()
        pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]
        adi_errs = []
        add_errs = []
        for i in range(len(pred_poses)):
            verts = meshes[obj_name].vertices.copy()
            adi = adi_err(pred_poses[i], gt_poses[i], verts)
            add = add_err(pred_poses[i], gt_poses[i], verts)
            adi_errs.append(adi)
            add_errs.append(add)

        adi_errs = np.array(adi_errs)
        add_errs = np.array(add_errs)
        adds_auc = compute_auc(adi_errs) * 100
        add_auc = compute_auc(add_errs) * 100
        total_eval_frames += len(pred_poses)

        per_object_results[obj_name] = {
            "pred_poses": pred_poses,
            "pred_poses_raw": pred_poses_raw,
            "gt_poses": gt_poses,
            "adi_errs": adi_errs,
            "add_errs": add_errs,
            "adds_auc": adds_auc,
            "add_auc": add_auc,
            "num_eval_frames": int(len(pred_poses)),
        }
        all_adi_errs.append(adi_errs)
        all_add_errs.append(add_errs)

        print(
            f"video {video_name}, obj {obj_name}, "
            f"ADD-S_err: {adi_errs.mean()*100:.2f}[cm], "
            f"ADD_errs: {add_errs.mean()*100:.2f}[cm], "
            f"ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}"
        )

        obj_eval_dir = os.path.join(out_folder, "evaluation", obj_name)
        os.makedirs(obj_eval_dir, exist_ok=True)
        plot_evaluation_results(
            add_s_errs=adi_errs,
            add_errs=add_errs,
            video_name=f"{video_name}_{obj_name}",
            output_dir=obj_eval_dir,
            save_plots=True,
            show_plots=False,
        )

        plot_error_over_time(
            add_s_errs=adi_errs,
            add_errs=add_errs,
            video_name=f"{video_name}_{obj_name}",
            output_dir=obj_eval_dir,
            save_plots=True,
            show_plots=False,
        )

        plot_pose_errors(
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            video_name=f"{video_name}_{obj_name}",
            output_dir=obj_eval_dir,
            save_plots=True,
            show_plots=False,
        )

        plot_pose_error_comparison(
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            video_name=f"{video_name}_{obj_name}",
            output_dir=obj_eval_dir,
            save_plots=True,
            show_plots=False,
        )

        plot_recall_vs_threshold(
            add_s_errs=adi_errs,
            add_errs=add_errs,
            video_name=f"{video_name}_{obj_name}",
            output_dir=obj_eval_dir,
            save_plots=True,
            show_plots=False,
            max_threshold=10.0,
        )

    mesh_paths = export_final_meshes_from_pipeline(pipeline, mesh_folder)

    if len(per_object_results) == 0:
        print("No GT poses found for any object, skipping evaluation.")
        return

    agg_adi = None
    agg_add = None
    if len(all_adi_errs) > 0:
        agg_adi = np.concatenate(all_adi_errs)
        agg_add = np.concatenate(all_add_errs)
        print(
            f"video {video_name}, aggregate over {len(per_object_results)} objects, "
            f"ADD-S_err: {agg_adi.mean()*100:.2f}[cm], "
            f"ADD_errs: {agg_add.mean()*100:.2f}[cm], "
            f"ADD-S_AUC: {compute_auc(agg_adi)*100:.2f}, "
            f"ADD_AUC: {compute_auc(agg_add)*100:.2f}"
        )

    mesh_cd_by_object = {}
    for obj_idx, obj_name in enumerate(object_names):
        if obj_name not in per_object_results:
            continue
        pred_mesh_path = mesh_paths.get(obj_idx, None)
        gt_visible_mesh_path = find_gt_visible_mesh_path(
            reader.video_dir, obj_name=obj_name
        )
        mesh_cd_cm = evaluate_reconstructed_mesh(
            pred_mesh_path=pred_mesh_path,
            gt_visible_mesh_path=gt_visible_mesh_path,
            output_dir=mesh_folder,
            mesh_prefix=f"{video_name}_{obj_name}",
            pred_pose_first=per_object_results[obj_name]["pred_poses_raw"][0],
            gt_pose_first=per_object_results[obj_name]["gt_poses"][0],
        )
        if np.isfinite(mesh_cd_cm):
            print(f"video {video_name}, obj {obj_name}, mesh_CD: {mesh_cd_cm:.3f}[cm]")
        else:
            print(f"video {video_name}, obj {obj_name}, mesh_CD: skipped")
        mesh_cd_by_object[obj_name] = mesh_cd_cm

    valid_mesh = [
        mesh_cd for mesh_cd in mesh_cd_by_object.values() if np.isfinite(mesh_cd)
    ]
    avg_mesh_cd = float(np.mean(valid_mesh)) if len(valid_mesh) > 0 else np.inf
    return {
        "video_name": video_name,
        "add_s_err_mean": (
            float(agg_adi.mean() * 100) if agg_adi is not None else np.inf
        ),
        "add_err_mean": float(agg_add.mean() * 100) if agg_add is not None else np.inf,
        "add_s_auc": float(compute_auc(agg_adi) * 100) if agg_adi is not None else 0.0,
        "add_auc": float(compute_auc(agg_add) * 100) if agg_add is not None else 0.0,
        "mesh_cd_cm": avg_mesh_cd,
        "num_eval_objects": int(len(per_object_results)),
        "num_eval_frames": int(total_eval_frames),
    }


def run_ycbinisaac_single(
    data_path: str, video_name: str, out_dir: str, config_path: str, model_path: str
):
    """Alias kept for consistent naming with the dataset and script filename."""
    return run_ycbineoat_single(data_path, video_name, out_dir, config_path, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults adapted for typical YCB structure or left generic
    parser.add_argument(
        "--data_path",
        "-d",
        default="/home/justin/data/test",
        type=str,
        help="Root directory containing video folders (e.g. /path/to/YCB_Video/data)",
    )
    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        default="021_bleach_cleanser_easy",
        help="Name of the video folder (e.g. 0048 or bleach0)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/ycbinisaac_single",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="./configs/ycbinisaac/ycbinisaac_single.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default="/home/justin/data/HO3D_V3/models",
        type=str,
        help="Root directory for YCB models (e.g. /path/to/YCB_Video/models)",
    )

    args = parser.parse_args()

    # Convert relative paths to absolute if needed, or rely on user providing valid paths

    run_ycbineoat_single(
        args.data_path, args.video_name, args.out_dir, args.config_path, args.model_path
    )
