"""Render the AP14 colored map at an exact viewer rotation (2D ortho scatter,
same convention as the interactive viewer: x right, y down, +z toward viewer)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = "/tmp/claude-1000/-home-justin-code-point-to-pose/98025e7b-880c-4111-94de-90330a4d52a2/scratchpad"
OUT = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"

R = np.array(json.loads(sys.argv[1]))
tag = sys.argv[2] if len(sys.argv) > 2 else "custom"
size = float(sys.argv[3]) if len(sys.argv) > 3 else 26.0

d = json.load(open(f"{SCRATCH}/map_points_AP14.json"))
p = np.array(d["pts"])
cols = np.array(d["rgb"]) / 255.0

q = p @ R.T
order = np.argsort(q[:, 2])            # far first, near last (painter's)
x, y = q[order, 0], -q[order, 1]       # canvas y-down -> plot y-up
c = cols[order]

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
ax.scatter(x, y, c=c, s=size, linewidths=0)
ax.set_aspect("equal")
ax.set_axis_off()
fig.tight_layout(pad=0)
out = f"{OUT}/map_colored_AP14_{tag}.png"
fig.savefig(out, transparent=True)
print("saved", out)
