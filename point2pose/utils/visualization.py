import os

import numpy as np
import cv2
import torch

from point2pose.utils.transform import to_homo


def draw_points_on_image(image, points, colors):
    # if points are tensor
    # print(points.shape)
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if isinstance(colors, np.ndarray):
        colors = colors.tolist()

    for i in range(points.shape[0]):
        cv2.circle(
            image,
            points[i, :].astype(int).reshape(2),
            radius=5,
            color=colors[i],
            thickness=-1,
        )


def get_n_uncertainty_colors(uncertainties, u_min=0.0, u_max=1.0, inverse=False):
    """
    Map uncertainties to BGR colors using OpenCV colormap (e.g., COLORMAP_JET).

    Args:
        uncertainties (np.ndarray): (N,) array of uncertainty values.
        u_min (float): minimum value for normalization.
        u_max (float): maximum value for normalization.
        inverse (bool): if True, invert the normalized scale.

    Returns:
        np.ndarray: (N, 3) array of colors in BGR (uint8).
    """
    # Normalize to [0,1]
    norm = np.clip((uncertainties - u_min) / (u_max - u_min + 1e-8), 0, 1)
    if inverse:
        norm = 1 - norm

    # Convert to 0–255 uint8 for colormap lookup
    norm_uint8 = (norm * 255).astype(np.uint8)

    # Apply OpenCV colormap (e.g., COLORMAP_JET)
    colors_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

    # Convert to plain array (N, 3)
    return colors_bgr.reshape(-1, 3)


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """
    Copied from FoundationPose.
    Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(
            int
        )  # (2,2)
        img = cv2.line(
            img,
            uv[0].tolist(),
            uv[1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(
    image,
    ob_in_cam,
    scale=0.1,
    K=np.eye(3),
    thickness=3,
    transparency=0,
    is_input_rgb=False,
):
    """
    Copied from FoundationPose.
    @color: BGR
    """
    if is_input_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = image.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp


def visualize_and_save_tracking_results(
    frame,
    objects,
    track_table,
    frame_id=None,
    visualize_points=True,
    points_vis_method="visible_uncertainty",
    save_images=False,
    output_image_dir=None,
    camera_intrinsics=np.eye(3),
    bbox_min_max=None,
):
    """
    Visualize tracking results on the frame
    TODO: make it more general by removing track_table dependency
    """
    display_frame = frame.rgb.copy()
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

    height, width = display_frame.shape[:2]

    # Draw segmentation masks if available
    if hasattr(frame, "mask") and frame.mask is not None:
        mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        mask_overlay[..., 1] = 255  # Green base

        for i in range(len(frame.mask)):
            obj_mask = frame.mask[i, 0] > 0.0
            ## TODO: optimize this by removing the cpu().numpy()
            obj_mask = obj_mask.cpu().numpy()
            if np.any(obj_mask):
                # Color each object differently
                hue = (i + 3) / (len(frame.mask) + 3) * 255
                mask_overlay[obj_mask, 0] = hue
                mask_overlay[obj_mask, 2] = 255

        mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_HSV2BGR)
        display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, 0.5, 0)

    if visualize_points:
        if points_vis_method == "uncertainty":

            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                uncertainty_color = get_n_uncertainty_colors(
                    track_table.uncertainty[track_table.obj2track_map[i]]
                )
                draw_points_on_image(
                    display_frame,
                    track_table.track_2d[track_table.obj2track_map[i]],
                    uncertainty_color,
                )
        elif points_vis_method == "visible":
            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                # Generate N by 3 array with (0,255,0) for each row
                track_2d_points = track_table.track_2d[track_table.obj2track_map[i]]
                N = len(track_2d_points)
                visible_color = np.full((N, 3), (0, 0, 255), dtype=np.uint8)
                visible_color[track_table.visible[track_table.obj2track_map[i]]] = (
                    0,
                    255,
                    0,
                )
                draw_points_on_image(
                    display_frame,
                    track_2d_points,
                    visible_color,
                )
        elif points_vis_method == "visible_uncertainty":
            # Plot only visible points, colored by their uncertainty colors
            for i, obj in enumerate(objects):
                if i not in track_table.obj2track_map:
                    continue

                track_idx = track_table.obj2track_map[i]
                track_2d_points = track_table.track_2d[track_idx]
                visible_mask = track_table.visible[track_idx]

                if np.any(visible_mask):
                    uncertainty_color = get_n_uncertainty_colors(
                        track_table.uncertainty[track_idx]
                    )

                    draw_points_on_image(
                        display_frame,
                        track_2d_points[visible_mask],
                        uncertainty_color[visible_mask],
                    )
        # elif points_vis_method == "frame_id":
        #     # Color each point based on the frame id it was first seen (object.key_point_frames)
        #     for i, obj in enumerate(objects):
        #         if i not in track_table.obj2track_map:
        #             continue

        #         track_idx = track_table.obj2track_map[i]
        #         track_2d_points = track_table.track_2d[track_idx]
        #         visible_mask = track_table.visible[track_idx]

        #         # Only proceed if there are visible points
        #         if not np.any(visible_mask):
        #             continue
        #         if obj.key_point_frames.shape[0] == 0:
        #             continue
        #         # Align per-object track order with object's key point order
        #         # Assume key_point_frames order corresponds to obj2track_map order
        #         num_tracks_for_obj = len(track_idx)
        #         kp_frames_for_obj = obj.key_point_frames[:num_tracks_for_obj].astype(
        #             np.int32
        #         )

        #         # Frame ids for visible points; replace unknown -1 with current frame id if available
        #         frame_ids = kp_frames_for_obj[visible_mask]
        #         if frame_id is not None:
        #             frame_ids = frame_ids.copy()
        #             frame_ids[frame_ids == -1] = int(frame_id)

        #         # Use cached, distinctive colors per frame id
        #         colors_bgr = self._colors_for_frame_ids(frame_ids)

        #         # Draw only visible points for this object, using aligned colors
        #         draw_points_on_image(
        #             display_frame,
        #             track_2d_points[visible_mask],
        #             colors_bgr,
        #         )

    # Draw pose information
    for i, obj in enumerate(objects):
        if obj.pose is not None:
            pose = obj.pose @ obj.init_pose
            if bbox_min_max is not None:
                bbox_min_max_local = bbox_min_max
            else:
                half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
                bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

            display_frame = draw_posed_3d_box(
                camera_intrinsics,
                display_frame,
                pose,
                bbox_min_max_local,
            )
            display_frame = draw_xyz_axis(
                image=display_frame, ob_in_cam=pose, K=camera_intrinsics
            )

    # Save image if flag is enabled and frame_id is provided
    if save_images and frame_id is not None:
        image_filename = os.path.join(output_image_dir, f"frame_{frame_id:06d}.png")
        cv2.imwrite(str(image_filename), display_frame)

    return display_frame


def visualize_and_save_tracking_results_with_gt(
    frame,
    objects,
    track_table,
    est_result_frame=None,
    gt_pose=None,
    frame_id=None,
    visualize_points=True,
    points_vis_method="visible_uncertainty",
    save_images=False,
    output_image_dir=None,
    camera_intrinsics=np.eye(3),
    bbox_min_max=None,
    gt_bbox_min_max=None,
    pred_pose_color=(0, 255, 0),  # Green for predicted pose
    gt_pose_color=(0, 0, 255),  # Red for GT pose
):
    """
    Visualize tracking results on the frame with ground truth pose and bounding box overlay.

    This function extends visualize_and_save_tracking_results by also drawing the GT pose
    and bounding box in a different color to allow comparison with the predicted pose.

    Args:
        frame: Frame object containing RGB image
        objects: List of tracked objects
        track_table: Track table containing point tracks
        gt_pose: Ground truth pose (4x4 transformation matrix) in camera frame. If None, GT won't be drawn.
        frame_id: Frame identifier for saving images
        visualize_points: Whether to visualize tracked points
        points_vis_method: Method for visualizing points ("uncertainty", "visible", "visible_uncertainty")
        save_images: Whether to save visualization images
        output_image_dir: Directory to save images
        camera_intrinsics: Camera intrinsic matrix (3x3)
        bbox_min_max: Bounding box min/max for predicted pose (2x3 array)
        gt_bbox_min_max: Bounding box min/max for GT pose (2x3 array). If None, uses bbox_min_max.
        pred_pose_color: BGR color for predicted pose visualization (default: green)
        gt_pose_color: BGR color for GT pose visualization (default: red)

    Returns:
        display_frame: Image with visualizations overlaid
    """
    # First, call the original function to get the base visualization
    if est_result_frame is not None:
        display_frame = est_result_frame
    else:
        display_frame = visualize_and_save_tracking_results(
            frame=frame,
            objects=objects,
            track_table=track_table,
            frame_id=None,  # Don't save yet, we'll save after adding GT
            visualize_points=visualize_points,
            points_vis_method=points_vis_method,
            save_images=False,  # Don't save yet
            output_image_dir=None,
            camera_intrinsics=camera_intrinsics,
            bbox_min_max=bbox_min_max,
        )

    # Now overlay GT pose and bounding box if provided
    if gt_pose is not None:
        # Use GT bbox if provided, otherwise use the same bbox as prediction
        if gt_bbox_min_max is not None:
            gt_bbox_min_max_local = gt_bbox_min_max
        elif bbox_min_max is not None:
            gt_bbox_min_max_local = bbox_min_max
        else:
            # Fallback: try to get bbox from first object
            if len(objects) > 0 and hasattr(objects[0], "bbox"):
                half = 0.5 * np.asarray(objects[0].bbox.extent, dtype=float)
                gt_bbox_min_max_local = np.vstack([-half, +half])  # (2,3)
            else:
                gt_bbox_min_max_local = None

        if gt_bbox_min_max_local is not None:
            # Draw GT bounding box in red
            display_frame = draw_posed_3d_box(
                camera_intrinsics,
                display_frame,
                gt_pose,
                gt_bbox_min_max_local,
                line_color=gt_pose_color,
                linewidth=2,
            )
            # Draw GT coordinate axes in red
            display_frame = draw_xyz_axis(
                image=display_frame,
                ob_in_cam=gt_pose,
                K=camera_intrinsics,
                thickness=3,
            )

    # # Re-draw predicted pose with specified color to ensure it's visible
    # # (in case GT was drawn on top)
    # for obj in objects:
    #     if obj.pose is not None:
    #         pose = obj.pose @ obj.init_pose
    #         if bbox_min_max is not None:
    #             bbox_min_max_local = bbox_min_max
    #         else:
    #             half = 0.5 * np.asarray(obj.bbox.extent, dtype=float)
    #             bbox_min_max_local = np.vstack([-half, +half])  # (2,3)

    #         # Re-draw predicted bounding box with specified color
    #         display_frame = draw_posed_3d_box(
    #             camera_intrinsics,
    #             display_frame,
    #             pose,
    #             bbox_min_max_local,
    #             line_color=pred_pose_color,
    #             linewidth=2,
    #         )
    #         # Re-draw predicted coordinate axes
    #         display_frame = draw_xyz_axis(
    #             image=display_frame,
    #             ob_in_cam=pose,
    #             K=camera_intrinsics,
    #             thickness=3,
    #         )

    # Save image if flag is enabled and frame_id is provided
    if save_images and frame_id is not None:
        image_filename = os.path.join(output_image_dir, f"frame_{frame_id:06d}.png")
        cv2.imwrite(str(image_filename), display_frame)

    return display_frame


# def colors_for_frame_ids(frame_ids: np.ndarray) -> np.ndarray:
#     """Return consistent BGR colors for the provided frame ids."""
#     if frame_ids.size == 0:
#         return np.empty((0, 3), dtype=np.uint8)

#     colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
#     unique_ids = np.unique(frame_ids.astype(np.int64))

#     for fid in unique_ids:
#         fid_int = int(fid)
#         if fid_int not in self._frame_color_lookup:
#             self._frame_color_lookup[fid_int] = generate_next_frame_color()

#         colors[frame_ids == fid] = self._frame_color_lookup[fid_int]

#     return colors


# def generate_next_frame_color():
#     """Generate a new distinctive HSV-based color and convert it to BGR."""
#     golden_ratio_conjugate = 0.6180339887498949
#     base_index = len(self._frame_color_lookup)
#     saturation_cycle = (255, 230, 200, 180)
#     value_cycle = (255, 235, 215)

#     attempt = 0
#     while True:
#         idx = base_index + attempt
#         hue = int(round(((idx * golden_ratio_conjugate) % 1.0) * 179)) % 180
#         saturation = saturation_cycle[idx % len(saturation_cycle)]
#         value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]

#         hsv_tuple = (hue, saturation, value)
#         if hsv_tuple not in self._frame_color_used_hsv:
#             self._frame_color_used_hsv.add(hsv_tuple)
#             hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
#             return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

#         attempt += 1
