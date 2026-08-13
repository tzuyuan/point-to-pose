import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Use non-interactive backend in headless runs.
try:
    if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
except (RuntimeError, ImportError):
    pass
import matplotlib.pyplot as plt


def get_all_video_names(data_path):
    """Get all valid YCBInIsaac sequence folder names from data_path."""
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset root directory not found: {data_path}")

    video_names = []
    for item in sorted(os.listdir(data_path)):
        item_path = os.path.join(data_path, item)
        if not os.path.isdir(item_path):
            continue

        rgb_path = os.path.join(item_path, "rgb")
        if not os.path.isdir(rgb_path):
            # jpg-only layout (e.g. YCBMultiTrack_recalib)
            rgb_path = os.path.join(item_path, "jpg")
        if not os.path.isdir(rgb_path):
            continue

        if len(os.listdir(rgb_path)) == 0:
            continue

        cam_k_path = os.path.join(item_path, "cam_K.txt")
        if not os.path.isfile(cam_k_path):
            continue

        video_names.append(item)

    return video_names


def _float_or_nan(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return np.nan


def _canonical_object_name(reader, obj_name: str) -> str:
    return reader.videoname_to_object.get(obj_name, obj_name)


def _load_ycb_mesh(reader, model_root: str, obj_name: str):
    """
    Load one object mesh from the model root.
    """
    ob_name = _canonical_object_name(reader, obj_name)
    candidates = [
        # os.path.join(model_root, ob_name, "textured.obj"),
        os.path.join(model_root, ob_name, "textured_simple.obj"),
        os.path.join(model_root, "models", ob_name, "textured_simple.obj"),
        os.path.join(model_root, ob_name, "google_16k", "textured.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            import trimesh

            return trimesh.load(path)
    raise FileNotFoundError(
        f"Could not find mesh for {ob_name} in {model_root}. Checked: {candidates}"
    )


def _load_binary_labels(
    reader,
    video_path: str,
    obj_name: str,
    label_root: str,
    label_file_name: str,
    num_frames: int,
    required: bool = True,
) -> Optional[np.ndarray]:
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
        if required:
            raise FileNotFoundError(
                f"Could not find {label_root}/{label_file_name} for object {obj_name}. "
                f"Checked: {candidates}"
            )
        return None

    labels = np.asarray(np.load(label_path)).reshape(-1) > 0
    if labels.shape[0] < num_frames:
        padded = np.zeros((num_frames,), dtype=bool)
        padded[: labels.shape[0]] = labels
        labels = padded
    elif labels.shape[0] > num_frames:
        labels = labels[:num_frames]
    return labels


def _ensure_frame_ids(frame_ids: Optional[np.ndarray], count: int) -> np.ndarray:
    if frame_ids is None:
        return np.arange(count, dtype=np.int64)
    arr = np.asarray(frame_ids).reshape(-1)
    out = np.arange(count, dtype=np.int64)
    n = min(count, arr.shape[0])
    for i in range(n):
        try:
            out[i] = int(arr[i])
        except Exception:
            out[i] = int(i)
    return out


def _extract_pose_from_entry(entry, obj_idx: int) -> Optional[np.ndarray]:
    if entry is None:
        return None

    if isinstance(entry, dict):
        keys_to_try = [obj_idx, str(obj_idx), f"obj_{obj_idx}"]
        for key in keys_to_try:
            if key in entry:
                return _extract_pose_from_entry(entry[key], 0)
        return None

    if isinstance(entry, (list, tuple)):
        if obj_idx < len(entry):
            return _extract_pose_from_entry(entry[obj_idx], 0)
        return None

    arr = np.asarray(entry)
    if arr.shape == (4, 4):
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        if obj_idx < arr.shape[0]:
            return arr[obj_idx].astype(np.float64)
        return None
    if arr.ndim == 3 and arr.shape[:2] == (4, 4):
        if obj_idx < arr.shape[2]:
            return arr[:, :, obj_idx].astype(np.float64)
        return None
    return None


def _extract_object_pose_series(
    poses_raw,
    frame_ids_raw: Optional[np.ndarray],
    obj_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      poses: (M,4,4)
      frame_ids: (M,)
    """
    poses = []
    frame_ids = []

    if isinstance(poses_raw, np.ndarray) and poses_raw.dtype != object:
        arr = np.asarray(poses_raw)

        if arr.ndim == 3:
            if arr.shape[1:] == (4, 4):
                if obj_idx != 0:
                    return np.empty((0, 4, 4), dtype=np.float64), np.empty(
                        (0,), dtype=np.int64
                    )
                ids = _ensure_frame_ids(frame_ids_raw, arr.shape[0])
                return arr.astype(np.float64), ids
            if arr.shape[:2] == (4, 4):
                if obj_idx != 0:
                    return np.empty((0, 4, 4), dtype=np.float64), np.empty(
                        (0,), dtype=np.int64
                    )
                arr_t = np.transpose(arr, (2, 0, 1))
                ids = _ensure_frame_ids(frame_ids_raw, arr_t.shape[0])
                return arr_t.astype(np.float64), ids

        if arr.ndim == 4 and arr.shape[-2:] == (4, 4):
            if frame_ids_raw is not None:
                n_frames_ref = np.asarray(frame_ids_raw).reshape(-1).shape[0]
            else:
                n_frames_ref = None

            if n_frames_ref is not None and arr.shape[0] == n_frames_ref:
                if obj_idx >= arr.shape[1]:
                    return np.empty((0, 4, 4), dtype=np.float64), np.empty(
                        (0,), dtype=np.int64
                    )
                ids = _ensure_frame_ids(frame_ids_raw, arr.shape[0])
                return arr[:, obj_idx].astype(np.float64), ids

            if n_frames_ref is not None and arr.shape[1] == n_frames_ref:
                if obj_idx >= arr.shape[0]:
                    return np.empty((0, 4, 4), dtype=np.float64), np.empty(
                        (0,), dtype=np.int64
                    )
                ids = _ensure_frame_ids(frame_ids_raw, arr.shape[1])
                return arr[obj_idx].astype(np.float64), ids

            # Fallback: treat first axis as frames and second as objects.
            if obj_idx < arr.shape[1]:
                ids = _ensure_frame_ids(frame_ids_raw, arr.shape[0])
                return arr[:, obj_idx].astype(np.float64), ids
            return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.int64)

        return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.int64)

    raw = np.asarray(poses_raw, dtype=object).reshape(-1)
    ids = _ensure_frame_ids(frame_ids_raw, raw.shape[0])
    for i, entry in enumerate(raw):
        pose = _extract_pose_from_entry(entry, obj_idx)
        if pose is None:
            continue
        if pose.shape != (4, 4):
            continue
        if not np.isfinite(pose).all():
            continue
        poses.append(pose.astype(np.float64))
        frame_ids.append(int(ids[i]))

    if len(poses) == 0:
        return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.asarray(poses, dtype=np.float64), np.asarray(frame_ids, dtype=np.int64)


def _load_pred_poses_from_metadata(
    meta_data_path: str,
    obj_idx: int,
    pose_key: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")

    data = np.load(meta_data_path, allow_pickle=True)
    frame_ids_raw = data["frame_id"] if "frame_id" in data else None
    pose_keys = (
        [pose_key]
        if pose_key != "auto"
        else ["obj_pose_all", "obj_pose", "pose_local", "pose_frontend"]
    )

    for key in pose_keys:
        if key not in data:
            continue
        poses, frame_ids = _extract_object_pose_series(
            data[key], frame_ids_raw, obj_idx
        )
        if len(poses) > 0:
            return poses, frame_ids

    available = sorted(data.files)
    raise KeyError(
        f"No usable pose series found for object index {obj_idx} in {meta_data_path}. "
        f"Tried keys={pose_keys}. Available keys include: {available[:12]}"
    )


def evaluate_sequence_from_metadata(
    data_path: str,
    video_name: str,
    out_dir: str,
    model_path: str,
    pose_key: str = "auto",
    require_all_objects: bool = True,
    skip_mesh_cd: bool = False,
):
    from point2pose.io.sources.dataset.datareader import YCBInIsaacReader
    from point2pose.utils.evaluation import add_err, adi_err, compute_auc
    from point2pose.utils.transform import inverse_SE3

    if not skip_mesh_cd:
        from point2pose.utils.mesh_eval import (
            evaluate_reconstructed_mesh,
            find_gt_visible_mesh_path,
        )

    video_path = os.path.join(data_path, video_name)
    result_dir = os.path.join(out_dir, video_name)
    meta_data_path = os.path.join(result_dir, "meta_data", "meta_data.npz")
    mesh_dir = os.path.join(result_dir, "mesh")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")

    reader = YCBInIsaacReader(video_path)
    object_names = reader.get_object_names()
    if len(object_names) == 0:
        raise RuntimeError(f"No objects found for sequence: {video_name}")

    # Guard against truncated runs: a crashed pipeline still saves metadata via
    # its finally-block, which would otherwise be scored as a (misleadingly
    # easy) short sequence.
    _meta = np.load(meta_data_path, allow_pickle=True)
    _n_meta = len(np.atleast_1d(_meta["frame_id"]))
    if _n_meta < len(reader) - 5:
        raise RuntimeError(
            f"Metadata truncated for {video_name}: {_n_meta}/{len(reader)} "
            "frames — the run likely crashed; rerun the sequence."
        )

    expected_num_objects = len(object_names)
    per_object_results = {}
    all_adi_errs = []
    all_add_errs = []
    mesh_cd_by_object = {}
    total_eval_frames = 0
    skipped_objects = []

    for obj_idx, obj_name in enumerate(object_names):
        try:
            pred_poses_raw, pred_frame_ids = _load_pred_poses_from_metadata(
                meta_data_path=meta_data_path,
                obj_idx=obj_idx,
                pose_key=pose_key,
            )
        except KeyError as exc:
            print(f"[{video_name}/{obj_name}] {exc}")
            skipped_objects.append(
                {"object_name": obj_name, "reason": f"pose series missing: {exc}"}
            )
            continue
        if len(pred_poses_raw) == 0:
            print(
                f"[{video_name}/{obj_name}] No predicted poses in metadata for obj_idx={obj_idx}, skipping."
            )
            skipped_objects.append(
                {
                    "object_name": obj_name,
                    "reason": f"empty predicted pose series for obj_idx={obj_idx}",
                }
            )
            continue

        labels_in_image = _load_binary_labels(
            reader=reader,
            video_path=video_path,
            obj_name=obj_name,
            label_root="is_obj_in_image_labels",
            label_file_name="is_obj_in_image.npy",
            num_frames=len(reader),
            required=False,
        )
        labels_mask_visible = _load_binary_labels(
            reader=reader,
            video_path=video_path,
            obj_name=obj_name,
            label_root="is_mask_visible",
            label_file_name="is_mask_visible.npy",
            num_frames=len(reader),
            required=False,
        )
        eval_mask = np.ones((len(reader),), dtype=bool)
        missing_label_files = []
        if labels_in_image is not None:
            eval_mask &= labels_in_image
        else:
            missing_label_files.append("is_obj_in_image_labels/is_obj_in_image.npy")
        if labels_mask_visible is not None:
            eval_mask &= labels_mask_visible
        else:
            missing_label_files.append("is_mask_visible/is_mask_visible.npy")

        if len(missing_label_files) > 0:
            print(
                f"[{video_name}/{obj_name}] Missing visibility labels: "
                f"{', '.join(missing_label_files)}. "
                "Evaluating all GT poses for missing criteria."
            )

        pred_eval = []
        gt_eval = []
        for pose, frame_id in zip(pred_poses_raw, pred_frame_ids):
            fid = int(frame_id)
            if fid < 0 or fid >= len(reader):
                continue
            if not eval_mask[fid]:
                continue
            gt_pose = reader.get_gt_pose(fid, obj_name=obj_name)
            if gt_pose is None:
                continue
            if not np.isfinite(pose).all():
                continue
            pred_eval.append(np.asarray(pose, dtype=np.float64))
            gt_eval.append(np.asarray(gt_pose, dtype=np.float64))

        if len(pred_eval) == 0:
            print(
                f"[{video_name}/{obj_name}] No valid frames after applying available visibility filtering."
            )
            skipped_objects.append(
                {
                    "object_name": obj_name,
                    "reason": "no valid eval frames after filtering",
                }
            )
            continue

        pred_eval = np.asarray(pred_eval, dtype=np.float64)
        gt_eval = np.asarray(gt_eval, dtype=np.float64)
        pred_eval_raw = pred_eval.copy()
        pred_eval_aligned = pred_eval @ inverse_SE3(pred_eval[0]) @ gt_eval[0]

        mesh = _load_ycb_mesh(reader, model_path, obj_name)
        verts = np.asarray(mesh.vertices, dtype=np.float64)

        adi_errs = np.asarray(
            [
                adi_err(pred_eval_aligned[i], gt_eval[i], verts)
                for i in range(len(gt_eval))
            ],
            dtype=np.float64,
        )
        add_errs = np.asarray(
            [
                add_err(pred_eval_aligned[i], gt_eval[i], verts)
                for i in range(len(gt_eval))
            ],
            dtype=np.float64,
        )

        adds_auc = float(compute_auc(adi_errs) * 100.0)
        add_auc = float(compute_auc(add_errs) * 100.0)
        total_eval_frames += len(gt_eval)

        if skip_mesh_cd:
            mesh_cd_cm = np.inf
        else:
            pred_mesh_path = os.path.join(mesh_dir, f"pred_mesh_obj_{obj_idx}.ply")
            if not os.path.exists(pred_mesh_path):
                pred_mesh_path = None
            gt_visible_mesh_path = find_gt_visible_mesh_path(
                reader.video_dir, obj_name=obj_name
            )
            mesh_cd_cm = evaluate_reconstructed_mesh(
                pred_mesh_path=pred_mesh_path,
                gt_visible_mesh_path=gt_visible_mesh_path,
                output_dir=mesh_dir,
                mesh_prefix=f"{video_name}_{obj_name}",
                pred_pose_first=pred_eval_raw[0],
                gt_pose_first=gt_eval[0],
            )

        per_object_results[obj_name] = {
            "add_s_err_cm": float(adi_errs.mean() * 100.0),
            "add_err_cm": float(add_errs.mean() * 100.0),
            "add_s_auc": float(adds_auc),
            "add_auc": float(add_auc),
            "mesh_cd_cm": float(mesh_cd_cm),
            "num_eval_frames": int(len(gt_eval)),
        }
        all_adi_errs.append(adi_errs)
        all_add_errs.append(add_errs)
        mesh_cd_by_object[obj_name] = mesh_cd_cm

        mesh_cd_str = (
            f"{mesh_cd_cm:.3f}[cm]" if np.isfinite(mesh_cd_cm) else "skipped"
        )
        print(
            f"{video_name}, obj {obj_name}, "
            f"ADD-S_err: {adi_errs.mean()*100:.2f}[cm], "
            f"ADD_err: {add_errs.mean()*100:.2f}[cm], "
            f"ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}, "
            f"mesh_CD: {mesh_cd_str}"
        )

    if len(per_object_results) == 0:
        raise RuntimeError(
            f"No evaluable objects found in metadata for sequence {video_name}."
        )
    if require_all_objects and len(per_object_results) != expected_num_objects:
        skipped_summary = ", ".join(
            f"{item['object_name']} ({item['reason']})" for item in skipped_objects
        )
        raise RuntimeError(
            f"Sequence {video_name}: benchmarked {len(per_object_results)}/"
            f"{expected_num_objects} objects. Missing objects: {skipped_summary}"
        )

    agg_adi = np.concatenate(all_adi_errs)
    agg_add = np.concatenate(all_add_errs)
    valid_mesh = [v for v in mesh_cd_by_object.values() if np.isfinite(v)]
    avg_mesh_cd = float(np.mean(valid_mesh)) if len(valid_mesh) > 0 else np.inf

    return {
        "video_name": video_name,
        "add_s_err_mean": float(agg_adi.mean() * 100.0),
        "add_err_mean": float(agg_add.mean() * 100.0),
        "add_s_auc": float(compute_auc(agg_adi) * 100.0),
        "add_auc": float(compute_auc(agg_add) * 100.0),
        "mesh_cd_cm": avg_mesh_cd,
        "num_eval_objects": int(len(per_object_results)),
        "num_eval_frames": int(total_eval_frames),
        "per_object_metrics": per_object_results,
    }


def _save_tables(results, failures, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)

    per_sequence_csv = os.path.join(summary_dir, "per_sequence_metrics.csv")
    summary_csv = os.path.join(summary_dir, "summary_metrics.csv")
    failures_json = os.path.join(summary_dir, "failed_sequences.json")

    rows = sorted(results, key=lambda x: x["video_name"])
    with open(per_sequence_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "add_s_err_cm",
                "add_err_cm",
                "add_s_auc",
                "add_auc",
                "mesh_cd_cm",
                "num_eval_objects",
                "num_eval_frames",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_name": row["video_name"],
                    "add_s_err_cm": f"{float(row['add_s_err_mean']):.6f}",
                    "add_err_cm": f"{float(row['add_err_mean']):.6f}",
                    "add_s_auc": f"{float(row['add_s_auc']):.6f}",
                    "add_auc": f"{float(row['add_auc']):.6f}",
                    "mesh_cd_cm": (
                        f"{float(row['mesh_cd_cm']):.6f}"
                        if np.isfinite(row["mesh_cd_cm"])
                        else "inf"
                    ),
                    "num_eval_objects": int(row.get("num_eval_objects", 0)),
                    "num_eval_frames": int(row.get("num_eval_frames", 0)),
                }
            )

    def mean_of(key, finite_only=False):
        vals = [float(r[key]) for r in rows]
        if finite_only:
            vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    summary_row = {
        "num_sequences_success": len(rows),
        "num_sequences_failed": len(failures),
        "mean_add_s_err_cm": mean_of("add_s_err_mean"),
        "mean_add_err_cm": mean_of("add_err_mean"),
        "mean_add_s_auc": mean_of("add_s_auc"),
        "mean_add_auc": mean_of("add_auc"),
        "mean_mesh_cd_cm": mean_of("mesh_cd_cm", finite_only=True),
        "total_eval_objects": int(sum(r.get("num_eval_objects", 0) for r in rows)),
        "total_eval_frames": int(sum(r.get("num_eval_frames", 0) for r in rows)),
    }

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    with open(failures_json, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    return per_sequence_csv, summary_csv, failures_json


def _save_per_object_table(results, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)
    per_object_csv = os.path.join(summary_dir, "per_object_metrics.csv")

    rows = sorted(results, key=lambda x: x["video_name"])
    flat_rows = []
    for row in rows:
        video_name = row["video_name"]
        per_object = row.get("per_object_metrics", {})
        if not isinstance(per_object, dict):
            continue
        for obj_name, metrics in sorted(per_object.items(), key=lambda x: str(x[0])):
            if not isinstance(metrics, dict):
                continue
            flat_rows.append(
                {
                    "video_name": video_name,
                    "object_name": str(obj_name),
                    "add_s_err_cm": float(metrics.get("add_s_err_cm", np.nan)),
                    "add_err_cm": float(metrics.get("add_err_cm", np.nan)),
                    "add_s_auc": float(metrics.get("add_s_auc", np.nan)),
                    "add_auc": float(metrics.get("add_auc", np.nan)),
                    "mesh_cd_cm": float(metrics.get("mesh_cd_cm", np.nan)),
                    "num_eval_frames": int(metrics.get("num_eval_frames", 0)),
                }
            )

    with open(per_object_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "object_name",
                "add_s_err_cm",
                "add_err_cm",
                "add_s_auc",
                "add_auc",
                "mesh_cd_cm",
                "num_eval_frames",
            ],
        )
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(
                {
                    "video_name": row["video_name"],
                    "object_name": row["object_name"],
                    "add_s_err_cm": (
                        f"{row['add_s_err_cm']:.6f}"
                        if np.isfinite(row["add_s_err_cm"])
                        else "nan"
                    ),
                    "add_err_cm": (
                        f"{row['add_err_cm']:.6f}"
                        if np.isfinite(row["add_err_cm"])
                        else "nan"
                    ),
                    "add_s_auc": (
                        f"{row['add_s_auc']:.6f}"
                        if np.isfinite(row["add_s_auc"])
                        else "nan"
                    ),
                    "add_auc": (
                        f"{row['add_auc']:.6f}"
                        if np.isfinite(row["add_auc"])
                        else "nan"
                    ),
                    "mesh_cd_cm": (
                        f"{row['mesh_cd_cm']:.6f}"
                        if np.isfinite(row["mesh_cd_cm"])
                        else "inf"
                    ),
                    "num_eval_frames": int(row["num_eval_frames"]),
                }
            )

    return per_object_csv


def _plot_sequence_metrics(rows, summary_dir):
    if len(rows) == 0:
        return None

    names = [r["video_name"] for r in rows]
    x = np.arange(len(names))
    add_s_err = np.asarray([_float_or_nan(r["add_s_err_mean"]) for r in rows])
    add_err = np.asarray([_float_or_nan(r["add_err_mean"]) for r in rows])
    add_s_auc = np.asarray([_float_or_nan(r["add_s_auc"]) for r in rows])
    add_auc = np.asarray([_float_or_nan(r["add_auc"]) for r in rows])
    mesh_cd = np.asarray([_float_or_nan(r["mesh_cd_cm"]) for r in rows])
    eval_frames = np.asarray([int(r.get("num_eval_frames", 0)) for r in rows])

    fig_w = max(14, 0.6 * len(names))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10), constrained_layout=True)

    axes[0, 0].bar(x - 0.2, add_s_err, width=0.4, label="ADD-S error (cm)")
    axes[0, 0].bar(x + 0.2, add_err, width=0.4, label="ADD error (cm)")
    axes[0, 0].set_title("Per-sequence ADD/ADD-S Error")
    axes[0, 0].set_ylabel("cm")
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.2, add_s_auc, width=0.4, label="ADD-S AUC")
    axes[0, 1].bar(x + 0.2, add_auc, width=0.4, label="ADD AUC")
    axes[0, 1].set_title("Per-sequence ADD/ADD-S AUC")
    axes[0, 1].set_ylabel("AUC (%)")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].bar(x, mesh_cd, width=0.6, label="Mesh Chamfer distance (cm)")
    axes[1, 0].set_title("Per-sequence Mesh Chamfer Distance")
    axes[1, 0].set_ylabel("cm")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].bar(x, eval_frames, width=0.6, color="tab:green")
    axes[1, 1].set_title("Evaluated Frames Per Sequence")
    axes[1, 1].set_ylabel("frames")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")

    out_path = os.path.join(summary_dir, "sequence_metrics.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_metric_distributions(rows, summary_dir):
    if len(rows) == 0:
        return None

    add_s_err = np.asarray([_float_or_nan(r["add_s_err_mean"]) for r in rows])
    add_err = np.asarray([_float_or_nan(r["add_err_mean"]) for r in rows])
    add_s_auc = np.asarray([_float_or_nan(r["add_s_auc"]) for r in rows])
    add_auc = np.asarray([_float_or_nan(r["add_auc"]) for r in rows])
    mesh_cd = np.asarray([_float_or_nan(r["mesh_cd_cm"]) for r in rows])

    data = [
        add_s_err[np.isfinite(add_s_err)],
        add_err[np.isfinite(add_err)],
        add_s_auc[np.isfinite(add_s_auc)],
        add_auc[np.isfinite(add_auc)],
        mesh_cd[np.isfinite(mesh_cd)],
    ]
    labels = ["ADD-S err (cm)", "ADD err (cm)", "ADD-S AUC", "ADD AUC", "Mesh CD (cm)"]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.boxplot(data, patch_artist=True, labels=labels)
    ax.set_title("Distribution Across Sequences")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    out_path = os.path.join(summary_dir, "metric_distributions.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_auc_vs_chamfer(rows, summary_dir):
    if len(rows) == 0:
        return None

    add_s_auc = np.asarray([_float_or_nan(r["add_s_auc"]) for r in rows])
    mesh_cd = np.asarray([_float_or_nan(r["mesh_cd_cm"]) for r in rows])
    add_err = np.asarray([_float_or_nan(r["add_err_mean"]) for r in rows])

    valid = np.isfinite(add_s_auc) & np.isfinite(mesh_cd)
    if not np.any(valid):
        return None

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(
        add_s_auc[valid],
        mesh_cd[valid],
        c=add_err[valid],
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel("ADD-S AUC (%)")
    ax.set_ylabel("Mesh Chamfer distance (cm)")
    ax.set_title("ADD-S AUC vs Mesh Chamfer")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ADD error (cm)")

    out_path = os.path.join(summary_dir, "auc_vs_mesh_cd.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def benchmark_ycbinisaac_all(
    data_path: str,
    out_dir: str,
    config_path: str,
    model_path: str,
    summary_dir: str | None = None,
    pose_key: str = "auto",
    require_all_objects: bool = True,
    skip_mesh_cd: bool = False,
):
    video_names = get_all_video_names(data_path)
    print(f"Found {len(video_names)} videos to process: {video_names}")
    print("-" * 80)

    if summary_dir is None:
        summary_dir = os.path.join(out_dir, "benchmark_summary")

    results = []
    failures = []

    for idx, video_name in enumerate(video_names, 1):
        print(f"\n[{idx}/{len(video_names)}] Processing video: {video_name}")
        try:
            result = evaluate_sequence_from_metadata(
                data_path=data_path,
                video_name=video_name,
                out_dir=out_dir,
                model_path=model_path,
                pose_key=pose_key,
                require_all_objects=require_all_objects,
                skip_mesh_cd=skip_mesh_cd,
            )
            if result is not None:
                results.append(result)
            else:
                failures.append(
                    {"video_name": video_name, "error": "No result returned"}
                )
        except Exception as exc:
            print(f"Error processing video {video_name}: {exc}")
            failures.append({"video_name": video_name, "error": str(exc)})
            import traceback

            traceback.print_exc()

    rows = sorted(results, key=lambda x: x["video_name"])
    per_sequence_csv, summary_csv, failures_json = _save_tables(
        rows, failures, summary_dir
    )
    per_object_csv = _save_per_object_table(rows, summary_dir)
    plot_paths = [
        _plot_sequence_metrics(rows, summary_dir),
        _plot_metric_distributions(rows, summary_dir),
        _plot_auc_vs_chamfer(rows, summary_dir),
    ]
    plot_paths = [p for p in plot_paths if p is not None]

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Successful sequences: {len(rows)}")
    print(f"Failed sequences: {len(failures)}")
    if len(rows) > 0:
        avg_add_s_err = float(np.mean([r["add_s_err_mean"] for r in rows]))
        avg_add_err = float(np.mean([r["add_err_mean"] for r in rows]))
        avg_add_s_auc = float(np.mean([r["add_s_auc"] for r in rows]))
        avg_add_auc = float(np.mean([r["add_auc"] for r in rows]))
        valid_mesh = [r["mesh_cd_cm"] for r in rows if np.isfinite(r["mesh_cd_cm"])]
        avg_mesh_cd = float(np.mean(valid_mesh)) if len(valid_mesh) > 0 else np.nan
        print(
            f"Average ADD-S_err: {avg_add_s_err:.2f}[cm], "
            f"ADD_err: {avg_add_err:.2f}[cm], "
            f"ADD-S_AUC: {avg_add_s_auc:.2f}, "
            f"ADD_AUC: {avg_add_auc:.2f}, "
            f"mesh_CD: {avg_mesh_cd:.3f}[cm]"
        )

    print("-" * 80)
    print(f"Per-sequence table: {per_sequence_csv}")
    print(f"Per-object table:   {per_object_csv}")
    print(f"Summary table: {summary_csv}")
    print(f"Failures log: {failures_json}")
    if len(plot_paths) > 0:
        print("Plots:")
        for path in plot_paths:
            print(f"  - {path}")
    print("=" * 80)

    return {
        "results": rows,
        "failures": failures,
        "per_sequence_csv": per_sequence_csv,
        "per_object_csv": per_object_csv,
        "summary_csv": summary_csv,
        "failures_json": failures_json,
        "plot_paths": plot_paths,
    }


def benchmark_ycbinisaac_single(
    data_path: str,
    video_name: str,
    out_dir: str,
    config_path: str,
    model_path: str,
    summary_dir: str | None = None,
    pose_key: str = "auto",
    require_all_objects: bool = True,
    skip_mesh_cd: bool = False,
):
    """
    Benchmark one YCBInIsaac sequence from saved metadata.
    """
    if summary_dir is None:
        summary_dir = os.path.join(out_dir, video_name, "benchmark_summary")

    results = []
    failures = []

    print(f"Benchmarking single video: {video_name}")
    print("-" * 80)
    try:
        result = evaluate_sequence_from_metadata(
            data_path=data_path,
            video_name=video_name,
            out_dir=out_dir,
            model_path=model_path,
            pose_key=pose_key,
            require_all_objects=require_all_objects,
            skip_mesh_cd=skip_mesh_cd,
        )
        results.append(result)
    except Exception as exc:
        print(f"Error processing video {video_name}: {exc}")
        failures.append({"video_name": video_name, "error": str(exc)})
        import traceback

        traceback.print_exc()

    rows = sorted(results, key=lambda x: x["video_name"])
    per_sequence_csv, summary_csv, failures_json = _save_tables(
        rows, failures, summary_dir
    )
    per_object_csv = _save_per_object_table(rows, summary_dir)
    plot_paths = [
        _plot_sequence_metrics(rows, summary_dir),
        _plot_metric_distributions(rows, summary_dir),
        _plot_auc_vs_chamfer(rows, summary_dir),
    ]
    plot_paths = [p for p in plot_paths if p is not None]

    print("\n" + "=" * 80)
    print("SINGLE-SEQUENCE BENCHMARK SUMMARY")
    print("=" * 80)
    if len(rows) == 1:
        row = rows[0]
        mesh_cd_str = (
            f"{row['mesh_cd_cm']:.3f}[cm]"
            if np.isfinite(row["mesh_cd_cm"])
            else "skipped"
        )
        print(
            f"{row['video_name']}, "
            f"ADD-S_err: {row['add_s_err_mean']:.2f}[cm], "
            f"ADD_err: {row['add_err_mean']:.2f}[cm], "
            f"ADD-S_AUC: {row['add_s_auc']:.2f}, "
            f"ADD_AUC: {row['add_auc']:.2f}, "
            f"mesh_CD: {mesh_cd_str}"
        )
    else:
        print("No successful result for this sequence.")

    print("-" * 80)
    print(f"Per-sequence table: {per_sequence_csv}")
    print(f"Per-object table:   {per_object_csv}")
    print(f"Summary table: {summary_csv}")
    print(f"Failures log: {failures_json}")
    if len(plot_paths) > 0:
        print("Plots:")
        for path in plot_paths:
            print(f"  - {path}")
    print("=" * 80)

    return {
        "results": rows,
        "failures": failures,
        "per_sequence_csv": per_sequence_csv,
        "per_object_csv": per_object_csv,
        "summary_csv": summary_csv,
        "failures_json": failures_json,
        "plot_paths": plot_paths,
    }


def run_ycbinisaac_all(
    data_path: str,
    out_dir: str,
    config_path: str,
    model_path: str,
    summary_dir: str | None = None,
    pose_key: str = "auto",
    require_all_objects: bool = True,
    skip_mesh_cd: bool = False,
):
    """
    Backward-compatible alias.
    """
    return benchmark_ycbinisaac_all(
        data_path=data_path,
        out_dir=out_dir,
        config_path=config_path,
        model_path=model_path,
        summary_dir=summary_dir,
        pose_key=pose_key,
        require_all_objects=require_all_objects,
        skip_mesh_cd=skip_mesh_cd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/justin/data/YCBMultiTrack_new"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbmultitrackreal",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/home/justin/code/point-to-pose/configs/ycbinisaac/ycbinisaac_single.yaml",
        help="Unused in metadata-only benchmarking; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/home/justin/data/HO3D_V3/models",
        help="Root directory of YCB models.",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/ycbmultitrackreal/benchmark_summary",
        help="Directory for benchmark tables/plots (default: <out_dir>/benchmark_summary).",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default="auto",
        help="Pose key in metadata to use: auto, obj_pose_all, obj_pose, pose_local, pose_frontend.",
    )
    parser.add_argument(
        "--allow_partial_objects",
        action="store_true",
        help=(
            "Allow sequences where only a subset of objects can be evaluated. "
            "By default, the benchmark requires all objects in a sequence to be benchmarked."
        ),
    )
    parser.add_argument(
        "--skip_mesh_cd",
        action="store_true",
        help="Skip mesh Chamfer-distance evaluation for faster benchmarking.",
    )
    args = parser.parse_args()

    benchmark_ycbinisaac_all(
        data_path=args.data_path,
        out_dir=args.out_dir,
        config_path=args.config_path,
        model_path=args.model_path,
        summary_dir=args.summary_dir,
        pose_key=args.pose_key,
        require_all_objects=not args.allow_partial_objects,
        skip_mesh_cd=args.skip_mesh_cd,
    )
