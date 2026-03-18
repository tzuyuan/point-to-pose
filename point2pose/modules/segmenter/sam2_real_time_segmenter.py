import inspect
import numpy as np
import torch

from sam2.build_sam import build_sam2_camera_predictor

from point2pose.core.base_segmenter import Segmenter
from point2pose.core.module_registry import SEGMENTER

# use bfloat16 for the entire script (if CUDA available)
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@SEGMENTER.register_module("sam2")
class Sam2RealTimeSegmenter(Segmenter):
    def __init__(self, config):
        super().__init__(config)
        self.name = "sam2_real_time"
        self.device = config.get("device", "cpu")
        legacy_vos_optimized = config.get("vos_optimized", None)
        self.vos_inference = bool(
            config.get(
                "vos_inference",
                legacy_vos_optimized if legacy_vos_optimized is not None else False,
            )
        )
        self.apply_postprocessing = config.get("apply_postprocessing", True)

        model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        checkpoint = config.get(
            "checkpoint", "checkpoints/sam2.1/sam2.1_hiera_large.pt"
        )

        hydra_overrides_extra = []
        compile_flags = {
            "compile_memory_encoder": config.get("compile_memory_encoder", False),
            "compile_memory_attention": config.get("compile_memory_attention", False),
            "compile_prompt_encoder": config.get("compile_prompt_encoder", False),
            "compile_mask_decoder": config.get("compile_mask_decoder", False),
        }
        for key, enabled in compile_flags.items():
            if enabled:
                hydra_overrides_extra.append(f"++model.{key}=true")

        build_kwargs = {
            "device": self.device,
            "hydra_overrides_extra": hydra_overrides_extra,
            "apply_postprocessing": self.apply_postprocessing,
        }

        # Point2Pose exposes a single config knob, `vos_inference`.
        # If the local SAM2 builder still uses the older `vos_optimized` argument,
        # map the same value over for backward compatibility.
        sig = inspect.signature(build_sam2_camera_predictor)
        if "vos_inference" in sig.parameters:
            build_kwargs["vos_inference"] = self.vos_inference
        elif "vos_optimized" in sig.parameters:
            build_kwargs["vos_optimized"] = self.vos_inference

        self.predictor = build_sam2_camera_predictor(
            model_cfg,
            checkpoint,
            **build_kwargs,
        )

        self.input_points = []
        self.input_labels = []
        self.input_point_groups = []
        self.input_label_groups = []
        self.num_obj = 0
        self.tracking_started = False
        self.frame_count = 0

    def add_input_points(self, points, labels):
        """
        Add new points to be tracked.
        Args:
            points (List[List[int]]): List of points, each defined by [x, y].
            labels (List[int]): List of labels, each defined by 1 or 0.
                                1 means positive point, 0 means negative point.
        """
        if points is None or labels is None:
            return

        if self._has_grouped_prompt_format(points, labels):
            for point_group, label_group in zip(points, labels):
                self.input_points.extend(point_group)
                self.input_labels.extend(label_group)
                self._append_prompt_group(point_group, label_group)
            return

        self.input_points.extend(points)
        self.input_labels.extend(labels)
        self._append_flat_points_as_prompt_groups(points, labels)

    def initialize(self, image, mask=None):
        """
        Initialize the segmenter with the provided points or mask.
        """
        self.predictor.load_first_frame(image)
        self.num_obj = 0

        # Initialize from mask if provided
        if mask is not None:
            object_masks = self._extract_object_masks(mask)
            added_obj = 0

            for mask_idx, object_mask in enumerate(object_masks):
                # sample a few points within each object mask as prompt points
                y_indices, x_indices = np.where(object_mask > 0)
                if len(y_indices) == 0:
                    print(
                        f"[SAM2] Mask {mask_idx} is empty. Skipping object initialization."
                    )
                    continue

                num_points = 5
                if len(y_indices) > num_points:
                    indices = np.linspace(0, len(y_indices) - 1, num_points, dtype=int)
                else:
                    indices = np.arange(len(y_indices))

                points = np.stack(
                    [x_indices[indices], y_indices[indices]], axis=1
                ).astype(np.float32)
                labels = np.ones(len(points), dtype=np.int32)

                obj_id = self.num_obj + added_obj
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
                print(
                    f"[SAM2] Added object {obj_id} from mask {mask_idx} with {len(points)} sampled points."
                )
                added_obj += 1

            self.num_obj += added_obj
            if added_obj == 0:
                print("[SAM2] Mask provided but no valid object masks were found.")

        # Add points to predictor if any exist
        # Return True if at least one object is added.
        if len(self.input_points) > 0 or len(self.input_point_groups) > 0:
            self.tracking_started = self._add_all_points_to_predictor()
        else:
            self.tracking_started = self.num_obj > 0

        if not self.tracking_started:
            print(
                "[SAM2] No objects added. Please call add_input_points() or provide a mask."
            )

    def _extract_object_masks(self, mask):
        """
        Convert different mask formats into a list of 2D boolean masks (one per object).
        Supports:
            - torch.Tensor / np.ndarray
            - list/tuple of per-object masks
            - [N, 1, H, W], [N, H, W], [1, H, W], [H, W]
            - labeled [H, W] maps (0 = background, each non-zero value = one object)
        """
        if isinstance(mask, (list, tuple)):
            object_masks = []
            for m in mask:
                arr = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                object_masks.append(arr.astype(bool))
            return object_masks

        arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)

        if arr.ndim == 4:
            # Typical SAM/seg tensors: [N, 1, H, W]
            if arr.shape[1] == 1:
                return [arr[i, 0].astype(bool) for i in range(arr.shape[0])]
            # Fallback: collapse channel dimension if not singleton.
            collapsed = np.any(arr > 0, axis=1)
            return [collapsed[i].astype(bool) for i in range(collapsed.shape[0])]

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                return [arr[0].astype(bool)]
            if arr.shape[-1] == 1:
                return [arr[..., 0].astype(bool)]
            # Assume stacked object masks [N, H, W].
            return [arr[i].astype(bool) for i in range(arr.shape[0])]

        if arr.ndim == 2:
            # Labeled mask: each non-zero label denotes one object.
            unique_vals = np.unique(arr)
            non_zero_vals = unique_vals[unique_vals != 0]
            if non_zero_vals.size > 1 and arr.dtype != bool:
                return [(arr == v) for v in non_zero_vals]
            return [arr.astype(bool)]

        raise ValueError(
            f"Unsupported mask shape: {arr.shape}. Expected 2D, 3D, 4D, or list/tuple."
        )

    def segment(self, image):
        """
        Segment the image using SAM2 with the provided bounding boxes.

        Args:
            image (np.ndarray): The input RGB image.

        Returns:
            out_obj_ids (List[int]): List of object IDs.
            out_mask_logits (torch.Tensor): mask logits. [num_obj, 1, height, width]
        """
        out_obj_ids, out_mask_logits = self.predictor.track(image)
        self.frame_count += 1
        return out_obj_ids, out_mask_logits

    def _add_all_points_to_predictor(self) -> bool:
        """
        Add all points to SAM2 predictor.
        Return True if at least one object is added.
        """
        obj_id = self.num_obj
        start_obj_id = obj_id
        total_points = 0
        prompt_groups = list(zip(self.input_point_groups, self.input_label_groups))

        if not prompt_groups and len(self.input_points) > 0:
            self._append_flat_points_as_prompt_groups(self.input_points, self.input_labels)
            prompt_groups = list(zip(self.input_point_groups, self.input_label_groups))

        for current_points, current_labels in prompt_groups:
            if len(current_points) == 0:
                continue
            if not np.any(np.asarray(current_labels, dtype=np.int32) == 1):
                print(
                    "[SAM2] Skipping a prompt group without a positive point. "
                    "Each object needs at least one positive click."
                )
                continue

            self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                points=np.array(current_points, dtype=np.float32),
                labels=np.array(current_labels, dtype=np.int32),
            )
            obj_id += 1
            total_points += len(current_points)

        added_obj = obj_id - start_obj_id
        self.num_obj += added_obj
        if self.num_obj > 0:
            print(
                f"[SAM2] Added {added_obj} object(s) with {total_points} points; total objects: {self.num_obj}"
            )
            return True
        else:
            print(
                "[SAM2] No objects added. Please call add_input_points() to add prompt points before calling initialize()"
            )
            return False

    def _has_grouped_prompt_format(self, points, labels) -> bool:
        if not isinstance(points, (list, tuple)) or not isinstance(labels, (list, tuple)):
            return False
        if len(points) == 0 or len(labels) == 0:
            return False

        first_points = points[0]
        first_labels = labels[0]

        if isinstance(first_points, np.ndarray):
            points_grouped = first_points.ndim == 2
        elif isinstance(first_points, (list, tuple)):
            points_grouped = len(first_points) == 0 or isinstance(
                first_points[0], (list, tuple, np.ndarray)
            )
        else:
            points_grouped = False

        labels_grouped = isinstance(first_labels, (list, tuple, np.ndarray))
        return points_grouped and labels_grouped

    def _append_flat_points_as_prompt_groups(self, points, labels):
        current_points = []
        current_labels = []

        for point, label in zip(points, labels):
            label = int(label)
            if label == 1 and current_points:
                self._append_prompt_group(current_points, current_labels)
                current_points = []
                current_labels = []

            current_points.append(point)
            current_labels.append(label)

        if current_points:
            self._append_prompt_group(current_points, current_labels)

    def _append_prompt_group(self, point_group, label_group):
        if len(point_group) != len(label_group):
            raise ValueError("Prompt points and labels must have the same length.")
        if len(point_group) == 0:
            return

        normalized_points = np.asarray(point_group, dtype=np.float32).reshape(-1, 2)
        normalized_labels = np.asarray(label_group, dtype=np.int32).reshape(-1)

        self.input_point_groups.append(normalized_points.tolist())
        self.input_label_groups.append(normalized_labels.tolist())
