"""Panel 1: Input — RGB frame + SAM2 per-object masks + initial query points, depth inset."""
import cv2
import numpy as np

from common import Run, OBJ_COLORS_BGR, OUT_DIR, overlay_alpha, upscale

r = Run()
T = 0  # initialization frame with user-provided points
S = 2  # upscale factor

img = upscale(r.rgb(T), S)
canvas = img.copy()

masks = [r.mask(T, o) for o in range(3)]
for o, m in enumerate(masks):
    if m is None:
        continue
    mu = cv2.resize(m.astype(np.uint8), None, fx=S, fy=S,
                    interpolation=cv2.INTER_NEAREST) > 0
    canvas = overlay_alpha(canvas, OBJ_COLORS_BGR[o], mu, alpha=0.42)
    cnts, _ = cv2.findContours(mu.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, OBJ_COLORS_BGR[o], 3, lineType=cv2.LINE_AA)

# initial query points (first 75 tracks at t=0), white ring + object-color fill
pts = r.track2d(T) * S
ids = r.track_obj_ids()[: len(pts)]
for (x, y), o in zip(pts, ids):
    if not np.isfinite(x) or o < 0:
        continue
    cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(x), int(y)), 5, OBJ_COLORS_BGR[o], -1, cv2.LINE_AA)

# depth inset, bottom-left, turbo colormap
dep = r.depth(T).astype(np.float32)
dep[dep == 0] = np.nan
lo, hi = np.nanpercentile(dep, [2, 98])
dn = np.clip((dep - lo) / (hi - lo), 0, 1)
dn[~np.isfinite(dn)] = 0
dcol = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
ih, iw = 240, 320
inset = cv2.resize(dcol, (iw, ih))
H, W = canvas.shape[:2]
pad = 16
y0, x0 = H - ih - pad, pad
canvas[y0:y0 + ih, x0:x0 + iw] = inset
cv2.rectangle(canvas, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 3, cv2.LINE_AA)

cv2.imwrite(f"{OUT_DIR}/panel1_inputs.png", canvas)
print("saved", f"{OUT_DIR}/panel1_inputs.png")
