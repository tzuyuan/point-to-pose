TRACKER_RAGGED_FIELDS = {
    "track2d",
    "uncertainties",
    "visibles",
    "track3d",
    "valid_depth",
}

OBJECT_RAGGED_FIELDS = {
    "obj_key_points",
    "obj_uncertainties",
    "obj_valid",
    "obj_key_point_frames",
}

KEYFRAME_RAGGED_FIELDS = set()

REGISTRATION_RAGGED_FIELDS = {
    "reg_key_points_idx",
    "reg_key_points",
    "reg_prev3d",
    "reg_curr3d",
    "reg_inliers",
    "reg_residuals",
}

DENSE_RECOVERY_RAGGED_FIELDS = {
    "dense_recovery_inliers_before",
    "dense_recovery_residuals_before",
    "dense_recovery_inliers_after",
    "dense_recovery_residuals_after",
}

EXTRACT_RAGGED_FIELDS = {
    "extract_vis_obj_mask",
    "extract_val_obj_mask",
    "extract_uncer_obj_mask",
    "extract_valid_kp_mask",
    "extract_uncertainty_thres",
    "extract_obj_idx",
    "extract_inside_mask",
    "extract_finite_xy",
}

RAGGED_FIELDS: set[str] = (
    TRACKER_RAGGED_FIELDS
    | OBJECT_RAGGED_FIELDS
    | KEYFRAME_RAGGED_FIELDS
    | REGISTRATION_RAGGED_FIELDS
    | EXTRACT_RAGGED_FIELDS
    | DENSE_RECOVERY_RAGGED_FIELDS
)
