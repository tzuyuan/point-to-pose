import numpy as np


class RecoveryManager:
    """
    Manages recovery of lost objects by registering the current frame
    against past keyframes that share visible keypoints.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline.params

        # Minimum number of frames an object must be lost before attempting recovery
        self.max_lost_frames = self.pipeline_cfg.get("max_lost_frames", 5)

        # Minimum number of shared keypoints required to attempt registration with a keyframe
        self.min_recovery_overlap = self.pipeline_cfg.get("min_recovery_overlap", 20)

        # Counter to track how many consecutive frames an object has been lost
        # keys are obj_id, values are integer counts
        self.lost_counter = {}

    def update(self, frame, fe_result, objects, track_table, keyframes, register):
        """
        Attempt to recover lost objects.

        Args:
            frame: Current frame object.
            fe_result: FrontEndResult from the current step.
            objects: List of Object instances.
            track_table: PointTrackTable containing global track information.
            keyframes: Dictionary mapping obj_id to list of KeyFrame objects.
            register: The registration module instance.

        Returns:
            recovered: Dictionary of recovery stats for objects that were successfully recovered.
        """
        recovered = {}

        for obj in objects:
            obj_id = obj.id

            # If object is not lost, reset counter and skip
            if not getattr(obj, "lost", False):
                self.lost_counter[obj_id] = 0
                continue

            # Increment lost counter
            self.lost_counter[obj_id] = self.lost_counter.get(obj_id, 0) + 1

            # Only attempt recovery if lost for enough frames
            # (using max_lost_frames name from config, though it acts as a threshold)
            if self.lost_counter[obj_id] < self.max_lost_frames:
                continue

            kfs = keyframes.get(obj_id, [])
            if not kfs:
                continue

            # Find the best keyframe for recovery
            # We look for a keyframe that has the most keypoints currently visible/valid
            best_kf = None
            max_overlap = 0

            # Current frame data from fe_result
            # We use the global track indices to match with keyframe points
            cur_visibles = fe_result.visibles
            cur_valid = fe_result.track_valid

            for kf in kfs:
                # Indices of points in the keyframe
                kf_indices = kf.kp_track_indices

                # Check which of these are visible and valid in current frame
                # Note: We assume kf_indices are within bounds of cur_visibles/cur_valid
                # which should be true if track_table only grows

                # Create mask for current validity of these specific indices
                if len(kf_indices) == 0:
                    continue

                # Safe guard against index out of bound (though unlikely in normal operation)
                if kf_indices.max() >= len(cur_visibles):
                    continue

                valid_mask = cur_visibles[kf_indices] & cur_valid[kf_indices]
                overlap_count = np.sum(valid_mask)

                if overlap_count > max_overlap:
                    max_overlap = overlap_count
                    best_kf = kf

            # If we found a good keyframe with enough overlap
            if best_kf is not None and max_overlap >= self.min_recovery_overlap:

                # Perform registration
                T_rel, stats = self._align_frame_to_keyframe(
                    frame, fe_result, obj, best_kf, register
                )

                # Check if recovery was successful
                if T_rel is not None and self._is_good_recovery(stats):
                    print(
                        f"[RecoveryManager] Recovered Object {obj_id} using KeyFrame {best_kf.frame_id} (overlap: {max_overlap})"
                    )

                    # Update object pose
                    # T_rel is T_curr_kf (transform from KeyFrame to Current)
                    # So T_curr = T_rel @ T_kf
                    obj.pose = T_rel @ best_kf.pose

                    # Reset lost state
                    obj.lost = False
                    self.lost_counter[obj_id] = 0

                    recovered[obj_id] = {
                        "kf_id": best_kf.frame_id,
                        "stats": stats,
                        "overlap": max_overlap,
                    }

        return recovered

    def _align_frame_to_keyframe(self, frame, fe_result, obj, kf, register):
        """
        Align current frame to the selected keyframe using 3D-3D registration.
        """
        # 1. Find common points
        kf_indices = kf.kp_track_indices

        # Mask for points valid in CURRENT frame
        # We assume they are valid in KF (since they are keypoints of that KF)
        # We assume fe_result.visibles/track_valid align with track IDs
        curr_mask = fe_result.visibles[kf_indices] & fe_result.track_valid[kf_indices]

        # If not enough points, skip (double check, though handled in update)
        if np.sum(curr_mask) < 3:
            return None, {}

        # 2. Extract 3D points

        # Source: KeyFrame points (in KeyFrame's camera coordinates)
        # We filter kf points using the mask
        prev3d = kf.kp_3d_camera[curr_mask]

        # Target: Current frame points (in Current camera coordinates)
        # We look up the global 3D positions for the matching indices
        common_indices = kf_indices[curr_mask]
        curr3d = fe_result.track_3d[common_indices]

        # Weights: Use current frame uncertainties
        sigma_tgt = fe_result.uncertainties[common_indices]

        # 3. Run registration
        # register() usually computes T such that T @ source ~ target
        # So T aligns KeyFrame -> Current Frame
        T_rel, stats = register.register(prev3d, curr3d, sigma_tgt=sigma_tgt)

        return T_rel, stats

    def _is_good_recovery(self, stats):
        """
        Heuristic to check if registration result is acceptable.
        """
        # We can check inlier ratio or residual
        residuals = stats.get("residuals", [])
        inliers = stats.get("inliers", [])

        if len(residuals) == 0:
            return False

        # Use the pipeline's residual threshold if available via config,
        # otherwise hardcode a reasonable default or use stats
        mean_residual = (
            np.mean(residuals[inliers]) if np.any(inliers) else np.mean(residuals)
        )

        # Simple check: if mean residual is low enough.
        # The threshold could be passed from config, here we use a default/heuristic
        # Ideally this should match the tracker's residual threshold
        residual_thres = self.pipeline_cfg.get("recovery_residual_thres", 0.05)

        if mean_residual > residual_thres:
            return False

        # Also check inlier ratio?
        if len(inliers) > 0:
            inlier_ratio = np.sum(inliers) / len(inliers)
            if inlier_ratio < 0.3:  # Require at least 30% inliers
                return False

        return True
