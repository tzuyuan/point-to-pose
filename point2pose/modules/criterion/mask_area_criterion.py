import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import os
from datetime import datetime

from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


@CRITERION.register_module("mask_area")
class MaskAreaCriterion(SampleCriterion):
    """
    Mask area criterion checks if the mask area is too small.

    Configuration parameters:
    - fit_mode: "convex" or "bbox" - method to compute point region area
    - ratio_threshold: float - threshold for point_area/mask_area ratio
    - debug_viz: bool - enable debug visualization (default: False)
    - debug_dir: str - directory to save debug images (default: "debug/criterion")

    Debug visualization creates a 3-panel figure showing:
    1. Original RGB image
    2. RGB with mask overlay (red)
    3. RGB with tracked points, convex hull/bbox, and mask outline
    """

    def __init__(self, config):
        super().__init__()
        self.fit_mode = config.get("fit_mode", "convex")
        self.ratio_threshold = config.get("ratio_threshold", 0.5)
        self.debug_viz = config.get("debug_viz", False)
        self.debug_dir = config.get("debug_dir", "debug/criterion")

        if self.debug_viz:
            os.makedirs(self.debug_dir, exist_ok=True)

    def check_sample_criterion(self, context: CriterionContext, obj_id: int) -> bool:
        mask = context.frame.mask[obj_id, 0] > 0

        obj_idx = context.track_table.obj2track_map[obj_id]
        vis_obj = np.asarray(context.track_table.visible, dtype=bool)[obj_idx]

        idx = obj_idx[vis_obj]
        points = context.track_table.track_2d[idx]

        # remove points outside of the mask
        original_point_count = points.shape[0]
        if points.shape[0] > 0:
            # Convert mask to numpy for indexing
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

            # Get image dimensions
            h, w = mask_np.shape

            # Filter points that are within image bounds and inside the mask
            valid_points_mask = (
                (points[:, 0] >= 0)
                & (points[:, 0] < w)  # x within bounds
                & (points[:, 1] >= 0)
                & (points[:, 1] < h)  # y within bounds
                & mask_np[
                    points[:, 1].astype(int), points[:, 0].astype(int)
                ]  # inside mask
            )

            # Keep only valid points
            points = points[valid_points_mask]
            idx = idx[valid_points_mask]

            # Debug info
            filtered_count = original_point_count - points.shape[0]
            if filtered_count > 0:
                print(
                    f"Filtered out {filtered_count} points outside mask (kept {points.shape[0]}/{original_point_count})"
                )

        # --- check mask area ---
        mask_area = torch.count_nonzero(mask)
        # mask area is 0 means object is not visible
        if mask_area == 0:
            return False

        if points.shape[0] < 3:
            return False  # degenerate case, trivially small

        # --- compute point region area ---
        if self.fit_mode == "convex":
            try:
                hull = ConvexHull(points)
                point_area = hull.volume  # in 2D, `volume` is the polygon area

                context.frame.convex_hull_xy = hull.points[hull.vertices]
            except (ValueError, RuntimeError):
                print("convex hull fails (e.g. collinear points)")
                return True  # convex hull fails (e.g. collinear points)
        elif self.fit_mode == "bbox":
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            point_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        else:
            raise ValueError("mode must be 'convex' or 'bbox'")

        # --- ratio ---
        ratio = point_area / mask_area

        print(f"point_area: {point_area}, mask_area: {mask_area}, ratio: {ratio}")

        result = ratio < self.ratio_threshold

        # Debug visualization if enabled
        if self.debug_viz:
            # Pass the original mask for visualization (before > 0 operation)
            original_mask = context.frame.mask[obj_id, 0]
            self._debug_visualize(context, obj_id, original_mask, points, ratio, result)

        return result

    def _debug_visualize(
        self,
        context: CriterionContext,
        obj_id: int,
        mask: torch.Tensor,
        points: np.ndarray,
        ratio: float,
        result: bool,
    ):
        """
        Create debug visualization showing points, convex hull, and mask overlay.
        """

        # Get RGB image
        rgb = context.frame.rgb.copy()
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        # Convert mask to numpy
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Original RGB image
        axes[0].imshow(rgb)
        axes[0].set_title(f"Original RGB (Frame {context.frame.id})")
        axes[0].axis("off")

        # Plot 2: RGB with mask overlay
        axes[1].imshow(rgb)
        # Overlay mask in red with transparency
        mask_overlay = np.zeros_like(rgb)
        mask_overlay[:, :, 0] = mask_np  # Red channel
        axes[1].imshow(mask_overlay, alpha=0.5)
        axes[1].set_title(f"RGB + Mask Overlay (Obj {obj_id})")
        axes[1].axis("off")

        # Plot 3: Points and convex hull
        axes[2].imshow(rgb)

        if points.shape[0] >= 3:
            # Plot tracked points
            axes[2].scatter(
                points[:, 0],
                points[:, 1],
                c="yellow",
                s=50,
                marker="o",
                edgecolors="black",
                linewidth=1,
                label=f"Tracked Points ({points.shape[0]})",
            )

            # Plot convex hull if possible
            if self.fit_mode == "convex":
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack(
                        [hull_points, hull_points[0]]
                    )  # Close the polygon
                    axes[2].plot(
                        hull_points[:, 0],
                        hull_points[:, 1],
                        "g-",
                        linewidth=2,
                        label="Convex Hull",
                    )
                except Exception as e:
                    axes[2].text(
                        0.02,
                        0.98,
                        f"Convex Hull Failed: {str(e)}",
                        transform=axes[2].transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
                    )
            elif self.fit_mode == "bbox":
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                    label="Bounding Box",
                )
                axes[2].add_patch(rect)
        else:
            axes[2].text(
                0.5,
                0.5,
                f"Insufficient Points ({points.shape[0]})",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

        # Overlay mask outline
        try:
            axes[2].contour(mask_np, levels=[0.5], colors="red", linewidths=2)
        except Exception:
            # If contour fails, just show a message
            pass
        axes[2].set_title(f"Points + Hull + Mask\nRatio: {ratio:.3f}, Result: {result}")
        axes[2].legend(loc="upper right")
        axes[2].axis("off")

        # Add overall title with statistics
        mask_area = (
            torch.count_nonzero(mask > 0)
            if isinstance(mask, torch.Tensor)
            else np.count_nonzero(mask)
        )
        point_area = self._compute_point_area(points)

        fig.suptitle(
            f"Mask Area Criterion Debug - Obj {obj_id}\n"
            f"Mask Area: {mask_area}, Point Area: {point_area:.1f}, "
            f"Ratio: {ratio:.3f}, Threshold: {self.ratio_threshold}, "
            f"Pass: {result}",
            fontsize=12,
        )

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"mask_area_debug_obj{obj_id}_frame{context.frame.id}_{timestamp}.png"
        )
        filepath = os.path.join(self.debug_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Debug visualization saved to: {filepath}")

    def _compute_point_area(self, points: np.ndarray) -> float:
        """Compute the area of the point region based on fit_mode."""
        if points.shape[0] < 3:
            return 0.0

        if self.fit_mode == "convex":
            try:
                hull = ConvexHull(points)
                return hull.volume  # in 2D, `volume` is the polygon area
            except (ValueError, RuntimeError):
                return 0.0
        elif self.fit_mode == "bbox":
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            return (x_max - x_min + 1) * (y_max - y_min + 1)
        else:
            return 0.0
