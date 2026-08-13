"""3D demo visualization plug-ins for the point2pose pipeline.

These modules are read-only observers of ModularPipeline state: attach them
to any runner with ``viz.update(pipeline, frame)`` — no pipeline changes.
"""

from point2pose.visualization.pose_3d_visualizer import Pose3DVisualizer
from point2pose.visualization.snapshot import ObjectSnapshot, SceneSnapshot

__all__ = ["Pose3DVisualizer", "SceneSnapshot", "ObjectSnapshot"]
