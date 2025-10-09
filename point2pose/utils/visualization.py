import numpy as np
import cv2
import torch
from point2pose.utils.transform import to_homo


def draw_points_on_image(image, points, colors):
    # if points are tensor
    # print(points.shape)
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    for i in range(points.shape[0]):
        cv2.circle(
            image,
            points[i, :].astype(int).reshape(2),
            radius=5,
            color=colors[i],
            thickness=-1,
        )


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
