import numpy as np
import sys
import os


import gtsam
from gtsam.symbol_shorthand import X, L

# Add the project root to the path
sys.path.append("/home/justin/code/point-to-pose")

from point2pose.utils.transform import transform_pts

# ---- 1) Load the whole npz ----
D = np.load(
    "/home/justin/code/point-to-pose/debug/pipeline/meta_data/meata_data.npz",
    allow_pickle=True,
)  # dict-like

print("Keys:", list(D.files))  # discover what's inside
N = len(D["frame_id"])  # number of rows/frames
print("Num rows:", N)


# ---- 2) Helper to unpack ragged fields ----
def unpack_ragged(name: str, store: dict, dim=-1):
    data = store[f"{name}_data"]
    offsets = store[f"{name}_offsets"]
    lengths = store[f"{name}_lengths"]
    out = []
    for off, L in zip(offsets, lengths):
        flat_data = data[off : off + L]
        # Reshape to (N, 3) assuming 3D points
        if dim == 3:
            reshaped_data = flat_data.reshape(-1, 3)
        elif dim == 2:
            reshaped_data = flat_data.reshape(-1, 2)
        elif dim == -1:
            reshaped_data = flat_data
        else:
            print(f"Warning: {name} data length {len(flat_data)} not divisible by 3")
            reshaped_data = flat_data  # Keep as 1D if can't reshape
        out.append(reshaped_data)
    return out  # -> list of (N, 3) ndarrays (one per row)


# ---- 3) Access fixed-shape fields (already stacked) ----
timestamp = D["timestamp"]  # shape (N,)
frame_id = D["frame_id"]  # shape (N,)

print(D["reg_key_points_data"].shape)

# ---- 4) Access ragged fields ----
reg_key_points_idx_list = unpack_ragged(
    "reg_key_points_idx", D
)  # list of (Mi,) int arrays
reg_key_points_list = unpack_ragged(
    "reg_key_points", D, dim=3
)  # list of (Mi,3) float arrays
reg_cur3d_list = unpack_ragged("reg_curr3d", D, dim=3)  # list of (Mi,3) float arrays
reg_inlier_list = unpack_ragged("reg_inliers", D)  # list of (Mi,) bool arrays
track3d = unpack_ragged("track3d", D, dim=3)
visibles = unpack_ragged("visibles", D)
uncertainties = unpack_ragged("uncertainties", D)


# obj pose
obj_init_pose = D["obj_init_pose"][0]
obj_pose_list = D["obj_pose"]
obj_key_points = unpack_ragged("obj_key_points", D, dim=3)
obj_uncertainties = unpack_ragged("obj_uncertainties", D)


print(f"\nExtracted registration data:")
print(f"  reg_key_points_list: {len(reg_key_points_list)} frames")
print(f"  reg_cur3d_list: {len(reg_cur3d_list)} frames")

# Show shapes for first few frames
# for i in range(min(3, len(reg_key_points_list))):
#     print(f"  Frame {i}: reg_key_points {reg_key_points_list[i].shape}, reg_cur3d {reg_cur3d_list[i].shape}")
#     print(reg_key_points_list[i])


# Create a register instance for debugging
config = {
    "debug_level": 1,
    "debug_dir": "/home/justin/code/point-to-pose/debug/register_test",
}

print(f"\nCreated SVDRegister instance with debug level: {config['debug_level']}")

# Store the data for use in other cells
print(f"\nData loaded successfully! Available variables:")
print(f"  - D: Full data dictionary")
print(f"  - N: Number of frames ({N})")
print(f"  - frame_id, obj_id, res_mean, num_points: Fixed-shape arrays")
print(f"  - reg_key_points_list, reg_cur3d_list: Lists of point clouds")


prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
between_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
)

graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

print(obj_init_pose)

X0 = gtsam.Pose3(obj_init_pose)
initial_estimate.insert(X(0), X0)

graph.push_back(gtsam.PriorFactorPose3(X(0), X0, prior_noise))

current_estimate = initial_estimate
for i in range(3):
    if i == 0:
        continue

    # add initial guess for the pose
    initial_estimate.insert(X(i), gtsam.Pose3(obj_pose_list[i]))

    # add between factor
    between_pose = gtsam.Pose3(obj_pose_list[i] @ np.linalg.inv(obj_pose_list[i - 1]))
    graph.push_back(
        gtsam.BetweenFactorPose3(X(i - 1), X(i), between_pose, between_noise)
    )


print("\nFactor Graph:\n{}".format(graph))


marginals = gtsam.Marginals(graph, current_estimate)
i = 0
while current_estimate.exists(X(i)):
    print(f"X{i} covariance:\n{marginals.marginalCovariance(X(i))}\n")
    i += 1
