"""Find full-occlusion -> re-entry events: per object, frames where the
visible-track fraction stays ~0 for a while and then jumps back."""
import sys

import numpy as np

from common import Run

for name in (sys.argv[1:] or ["ycb3", "ycb2h", "ycb3b"]):
    r = Run(name)
    ids = r.track_obj_ids_voted()
    n = r.n_frames
    frac = np.zeros((r.n_obj, n))
    for t in range(n):
        v = r.visibles(t)
        val = r.valid(t)
        k = len(v)
        for o in range(r.n_obj):
            sel = (ids[:k] == o) & val
            frac[o, t] = v[sel].mean() if sel.sum() > 10 else np.nan
    for o in range(r.n_obj):
        f = frac[o]
        occ = f < 0.06
        t = 0
        while t < n:
            if occ[t]:
                start = t
                while t < n and occ[t]:
                    t += 1
                dur = t - start
                if dur >= 25 and t < n - 10:
                    after = np.nanmax(f[t:t + 30])
                    before = np.nanmax(f[max(0, start - 30):start])
                    if after > 0.35 and before > 0.35:
                        print(f"{name} obj{o} ({r.obj_names[o]}): occluded "
                              f"{start}-{t} ({dur}f), before {before:.2f} "
                              f"after {after:.2f}")
            else:
                t += 1
