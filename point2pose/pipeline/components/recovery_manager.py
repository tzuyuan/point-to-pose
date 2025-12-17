import numpy as np
from scipy.spatial.transform import Rotation as R


class RecoveryManager:
    """
    Manages recovery of lost objects by registering the current frame
    against past keyframes that share visible keypoints.
    """

    def __init__(self, cfg):
        self.recovery_cfg = cfg.recovery.params

        # Minimum number of frames an object must be lost before attempting recovery
        self.max_lost_frames = self.recovery_cfg.get("max_lost_frames", 1)

        # Minimum number of shared keypoints required to attempt registration with a keyframe
        self.min_recovery_overlap = self.recovery_cfg.get("min_recovery_overlap", 10)

        # Clustering thresholds
        self.cluster_trans_thres = self.recovery_cfg.get(
            "cluster_trans_thres", 0.05
        )  # meters
        self.cluster_rot_thres = self.recovery_cfg.get(
            "cluster_rot_thres", 15.0
        )  # degrees

        self.num_reset = self.recovery_cfg.get("update_every", -1)

        print(f"[RecoveryManager] num_reset: {self.num_reset}")

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

            is_lost = getattr(obj, "lost", False)
            force_recovery = (self.num_reset > 0) and (frame.id % self.num_reset == 0)

            # If object is not lost, reset counter and skip
            if not is_lost and not force_recovery:
                self.lost_counter[obj_id] = 0
                continue

            # Increment lost counter
            if is_lost:
                self.lost_counter[obj_id] = self.lost_counter.get(obj_id, 0) + 1

            # Only attempt recovery if lost for enough frames
            # (using max_lost_frames name from config, though it acts as a threshold)
            if (
                not force_recovery
                and self.lost_counter.get(obj_id, 0) < self.max_lost_frames
            ):
                continue

            # get key frames for the object
            kfs = keyframes.get(obj_id, [])
            if not kfs:
                continue

            candidates = []

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

                # Filter by overlap count first
                if overlap_count > self.min_recovery_overlap:
                    # Perform registration
                    T_rel, stats = self._align_frame_to_keyframe(
                        frame, fe_result, obj, kf, register
                    )

                    # Check if registration was successful
                    if T_rel is not None and self._is_good_recovery(stats):
                        # Calculate candidate pose: T_curr = T_rel @ T_kf
                        pose_candidate = T_rel @ kf.pose
                        candidates.append(
                            {
                                "pose": pose_candidate,
                                "kf_id": kf.frame_id,
                                "stats": stats,
                                "overlap": overlap_count,
                            }
                        )

            # If we found valid candidates, cluster them
            if candidates:
                final_pose, info = self._cluster_and_select_pose(candidates)

                if final_pose is not None:
                    cluster_size = info.get("cluster_size", 1)
                    print(
                        f"[RecoveryManager] Recovered Object {obj_id} using cluster of {cluster_size} KeyFrames (best kf: {info['kf_id']})"
                    )

                    # Update object pose
                    obj.pose = final_pose

                    # Reset lost state
                    obj.lost = False
                    self.lost_counter[obj_id] = 0

                    recovered[obj_id] = {
                        "kf_id": info["kf_id"],
                        "stats": info["stats"],
                        "overlap": info["overlap"],
                        "cluster_size": cluster_size,
                    }

        return recovered

    def _cluster_and_select_pose(self, candidates):
        """
        Cluster candidate poses and return the average of the largest cluster.
        """
        if not candidates:
            return None, None

        if len(candidates) == 1:
            return candidates[0]["pose"], candidates[0]

        poses = [c["pose"] for c in candidates]
        n = len(poses)

        # Voting / Clustering
        # Count how many neighbors each pose has within threshold
        scores = np.zeros(n)
        adjacency = []  # Store neighbors for each index

        for i in range(n):
            neighbors = []
            p1 = poses[i]
            t1 = p1[:3, 3]
            R1 = p1[:3, :3]

            for j in range(n):
                p2 = poses[j]
                t2 = p2[:3, 3]
                R2 = p2[:3, :3]

                # Translation distance
                dt = np.linalg.norm(t1 - t2)

                # Rotation distance (angle)
                # trace(R1 @ R2.T) = 1 + 2cos(theta)
                R_diff = R1 @ R2.T
                tr = (np.trace(R_diff) - 1) / 2.0
                tr = np.clip(tr, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(tr))

                if dt < self.cluster_trans_thres and angle_deg < self.cluster_rot_thres:
                    neighbors.append(j)

            adjacency.append(neighbors)
            scores[i] = len(neighbors)

        # Pick cluster with max support
        best_idx = np.argmax(scores)
        cluster_indices = adjacency[best_idx]

        # Average the poses in the cluster
        cluster_poses = [poses[k] for k in cluster_indices]

        # 1. Average Translation
        mean_t = np.mean([p[:3, 3] for p in cluster_poses], axis=0)

        # 2. Average Rotation (via quaternions)
        quats = []
        for p in cluster_poses:
            q = R.from_matrix(p[:3, :3]).as_quat()
            quats.append(q)

        # Handle quaternion sign ambiguity (q and -q are same rotation)
        # Align all to the first one
        q0 = quats[0]
        for k in range(1, len(quats)):
            if np.dot(quats[k], q0) < 0:
                quats[k] = -quats[k]

        mean_q = np.mean(quats, axis=0)
        mean_q /= np.linalg.norm(mean_q)  # Normalize

        mean_R = R.from_quat(mean_q).as_matrix()

        final_pose = np.eye(4)
        final_pose[:3, :3] = mean_R
        final_pose[:3, 3] = mean_t

        # Use info from the "center" candidate (the one with most neighbors)
        # or just the representative
        best_info = candidates[best_idx].copy()
        best_info["cluster_size"] = len(cluster_indices)

        return final_pose, best_info

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
        try:
            T_rel, stats = register.register(prev3d, curr3d, sigma_tgt=sigma_tgt)
            print(T_rel)
        except Exception as e:
            print(f"[RecoveryManager] Error in aligning frame to keyframe: {e}")
            return None, {}

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
        residual_thres = self.recovery_cfg.get("recovery_residual_thres", 0.05)

        if mean_residual > residual_thres:
            return False

        # Also check inlier ratio?
        if len(inliers) > 0:
            inlier_ratio = np.sum(inliers) / len(inliers)
            if inlier_ratio < 0.3:  # Require at least 30% inliers
                return False

        return True
