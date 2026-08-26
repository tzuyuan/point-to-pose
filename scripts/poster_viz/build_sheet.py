"""Build the poster-module contact sheet HTML with embedded images."""
import base64
import io
import os

from PIL import Image

OUT = "/home/justin/results/eccv_point2pose/paper_figs/poster_modules"
SHEET = "/tmp/claude-1000/-home-justin-code-point-to-pose/98025e7b-880c-4111-94de-90330a4d52a2/scratchpad/poster_sheet.html"


def uri(name, width=900, quality=82):
    p = os.path.join(OUT, name)
    im = Image.open(p)
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


MODULES = [
    dict(
        num="0",
        title="Live demo still \u2014 straight from the rerun stream",
        maps_to="teaser / runtime corner (\u201cruns live at 2\u201310 Hz\u201d)",
        note="Frames extracted from cheetah.rrd with the live overlay exactly as the demo drew it (green box + tracked points). All 766 frames of the stream are recoverable from the .rrd, so any moment can become a still or filmstrip.",
        primary=("panel8_live_demo_f256.jpg", "live demo frame 256"),
        alts=[
            ("panel8_live_demo_f766.jpg", "live demo frame 766"),
        ],
    ),
    dict(
        num="1",
        title="Inputs — RGB-D + SAM2 masks + query points",
        maps_to="Fig. 2 “RGB / Segmentation Mask / Depth” inputs",
        note="Frame t=0 of the 3-object YCBMultiTrack-Real sequence. Per-object masks in the three object colors, initial query points as dots, depth inset bottom-left. One image replaces three small input thumbnails.",
        primary=("panel1_inputs.png", "t=0 · masks + queries + depth inset"),
        alts=[],
    ),
    dict(
        num="2",
        title="2D point tracker — long-range trails",
        maps_to="§3.1 “2D Point Tracker” box",
        note="CoTracker-style rendering of the run's real tracks: rainbow colors by spatial position, thick trails with an alpha ramp, dots with white rings. Single-object mode (obj N) lifts the caps and draws every visible point of one object. Any frame of any sequence (including a new recording) re-renders with panel_tracks.py.",
        primary=("panel2_tracks_ycb3_t1083_obj1.png", "t=1083 · SPAM can only, all points (recommended)"),
        alts=[
            ("panel2_tracks_ycb3_t1290_obj0.png", "t=1290 · mustard only, all points"),
            ("panel2_tracks_ycb3_t1290.png", "t=1290 · all objects, subsampled"),
            ("panel2_tracks_ho3d_AP14_t665.png", "HO3D AP14 · pitcher rotation"),
        ],
    ),
    dict(
        num="2c",
        title="Point sampling at keyframes",
        maps_to="\u00a73.2 sampling strategy \u2014 new query points on newly seen surface",
        note="A real sampling event, showing ONLY the 30 query points sampled at this keyframe as rainbow dots (same style as the trajectory figure), inside the mask contour. Sampling fires only when coverage drops \u2014 no event at t=550 itself because tracking was stable; nearest events are t=451 / 651 / 672. A no-contour variant of t=651 is also saved.",
        primary=("sampling_AP14_t651.png", "AP14 t=651 \u00b7 30 new samples (recommended)"),
        alts=[
            ("sampling_AP14_t451.png", "AP14 t=451 \u00b7 event before the stable stretch"),
            ("sampling_AP14_t672.png", "AP14 t=672 \u00b7 following event"),
        ],
    ),
    dict(
        num="2b",
        title="Occlusion \u2192 re-entry: tracks snap back",
        maps_to="new panel \u2014 the long-range-tracking claim, shown not told",
        note="Four frames, same rainbow colors throughout (color = identity): tracked \u2192 fully occluded (faded rings = last-seen configuration, identities held) \u2192 the same colors snap back onto the same object parts \u2192 resumed. The recommended strip is 448 dense TAPNext++ points on the new plush-cheetah recording; the gray timeline bands are four separate full occlusions \u2014 it relocalizes every time. Individual frames saved separately; full overlay video in dense_track_tapnext.mp4.",
        primary=("reloc_test_tapnext_25_strip.png", "NEW cheetah recording \u00b7 448 dense points, TAPNext++ \u2014 4 occlusion cycles in the timeline (recommended)"),
        alts=[
            ("reloc_test_tapnext_245_strip.png", "same take \u00b7 tight window around the last occlusion"),
            ("reentry_ycb3_obj1_745_strip.png", "YCB run \u00b7 SPAM behind arm, sparse pipeline points"),
            ("reentry_ycb3_obj1_1090_strip.png", "YCB run \u00b7 out of view 93 frames"),
        ],
    ),
    dict(
        num="3",
        title="Object-centric keypoint map",
        maps_to="§3.2 “Sampling Strategy and Keypoints Map” box",
        note="The tomato-soup-can map from the run, colored by the frame each keypoint was promoted (viridis: dark = early, bright = late). The growth strip makes the “map grows online at keyframes” story visual — this also covers the sampling module without a separate panel.",
        primary=("panel3_map_final_v1.png", "final map · front view"),
        alts=[
            ("panel3_map_growth.png", "growth strip t=60 / 600 / 1586"),
            ("panel3_map_final_v2.png", "final map · 3/4 view"),
        ],
    ),
    dict(
        num="4",
        title="Multi-hypothesis registration",
        maps_to="§3.3 “Frame-to-Map Registration” + pose-hypothesis box",
        note="Real RANSAC hypotheses from the run at a grasp moment: green solid box = hypothesis selected by TSDF consistency (green dots = its inlier map points), red dashed = the strongest rejected alternatives. Shows exactly why single-hypothesis RANSAC fails on symmetric cans.",
        primary=("panel4_hypotheses_t671.png", "t=671 · grasp ambiguity (recommended)"),
        alts=[
            ("panel4_hypotheses_t1035.png", "t=1035 · two-hand scene"),
            ("panel4_hypotheses_t1410.png", "t=1410 · under occlusion"),
        ],
    ),
    dict(
        num="4b",
        title="SVD registration \u2014 correspondences",
        maps_to="\u00a73.3 frame-to-map registration (the SVD step itself)",
        note="Real correspondences from the run at t=431: green filled = observed 3D points (current frame), amber rings = map keypoints projected under the previous pose, lines = the residual the weighted SVD closes. The 3D version shows the same data as two point clouds in their native frames \u2014 green lines = inliers, pink = rejected outliers (note the drifted cluster the outlier gate catches).",
        primary=("svdreg_img_t431.png", "t=431 \u00b7 on-image, can in hand (recommended)"),
        alts=[
            ("svdreg_3d_t431.png", "t=431 \u00b7 two-cloud 3D diagram"),
            ("svdreg_img_t401.png", "t=401 \u00b7 on-image alternative"),
            ("svdreg_3d_t1353.png", "t=1353 \u00b7 3D diagram, with full map context"),
        ],
    ),
    dict(
        num="5",
        title="Pose-graph optimization",
        maps_to="§3.4 “Graph Optimization” box",
        note="From the cheetah.rrd live-demo recording: real keyframe camera poses with their actual keyframe images as floating planes around the colored keypoint map; thin gray lines = keypoint observations. Everything is real logged data \u2014 poses, images, map, colors.",
        primary=("panel5_graph_cheetah.png", "cheetah live demo · real keyframe images (recommended)"),
        alts=[
            ("panel5_graph.png", "YCB run · schematic frusta"),
        ],
    ),
    dict(
        num="6",
        title="TSDF reconstruction",
        maps_to="§3.5 “3D Reconstruction” + TSDF box",
        note="The growth strip (from cheetah.rrd) shows the TSDF mesh assembling live as the plush toy is rotated in hand \u2014 the online-reconstruction story in one image. Single-object PLY renders as alternates; cans reconstruct less cleanly.",
        primary=("panel6_mesh_growth_cheetah.png", "cheetah plush · TSDF growth over the live demo (recommended)"),
        alts=[
            ("panel6_mesh_mustard_a.png", "mustard bottle · pred_mesh"),
            ("panel6_mesh_sloth_a.png", "sloth (demo object)"),
            ("panel6_mesh_bleach_a.png", "bleach cleanser"),
            ("panel6_mesh_ap14_a.png", "HO3D pitcher (textured)"),
        ],
    ),
    dict(
        num="7",
        title="Pose output — tracked 6D boxes",
        maps_to="Fig. 2 “Pose Output” box",
        note="Oriented 3D boxes + axis triads projected under the tracked pose. Each box is the OBB of the object's t=0 masked depth cloud (the pipeline's own init-bbox construction), so it stays rigidly attached to the object — fully model-free, no CAD. t=1500 shows two objects mid-air.",
        primary=("panel7_output_t1500.png", "t=1500 · objects in hand (recommended)"),
        alts=[
            ("panel7_output_t700.png", "t=700 · all three on table"),
        ],
    ),
]

CSS = """
:root{
  --bg:#fbfaf7; --card:#ffffff; --ink:#20242a; --muted:#6a7280;
  --line:#e2ddd2; --amber:#c77f00; --amber-soft:#fff3dc;
  --chip-ink:#7a5200;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#191b1f; --card:#22252b; --ink:#e8e6e1; --muted:#9aa1ab;
    --line:#33373f; --amber:#ffb000; --amber-soft:#3a2f14; --chip-ink:#ffd479;
  }
}
:root[data-theme="dark"]{
  --bg:#191b1f; --card:#22252b; --ink:#e8e6e1; --muted:#9aa1ab;
  --line:#33373f; --amber:#ffb000; --amber-soft:#3a2f14; --chip-ink:#ffd479;
}
*{box-sizing:border-box}
body{background:var(--bg); color:var(--ink);
  font:16px/1.55 "Avenir Next","Segoe UI",system-ui,sans-serif;
  margin:0; padding:2.2rem 1.2rem 4rem}
main{max-width:1040px; margin:0 auto}
header h1{font-size:1.7rem; line-height:1.2; margin:0 0 .3rem; text-wrap:balance}
header p.sub{color:var(--muted); margin:0 0 2rem; max-width:64ch}
.eyebrow{font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--amber); font-weight:700; margin-bottom:.4rem}
section.mod{border-top:1px solid var(--line); padding:1.6rem 0 2rem}
.modhead{display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap}
.modnum{font-weight:800; color:var(--amber); font-variant-numeric:tabular-nums}
.modhead h2{font-size:1.18rem; margin:0}
.maps{color:var(--muted); font-size:.85rem; margin:.15rem 0 .8rem}
.note{max-width:70ch; margin:.2rem 0 1rem; color:var(--ink)}
figure{margin:0}
figure img{max-width:100%; height:auto; display:block; border-radius:6px;
  border:1px solid var(--line); background:#fff}
figcaption{font-size:.8rem; color:var(--muted); margin-top:.35rem}
.pick{display:inline-block; background:var(--amber-soft); color:var(--chip-ink);
  font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:.15rem .5rem; border-radius:4px; margin-left:.5rem}
.alts{display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:.9rem; margin-top:1rem}
.plan{background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:1.3rem 1.5rem; margin:0 0 2.2rem}
.plan h2{margin:0 0 .8rem; font-size:1.15rem}
.plan table{border-collapse:collapse; width:100%; font-size:.92rem}
.plan td, .plan th{padding:.45rem .6rem; border-top:1px solid var(--line);
  text-align:left; vertical-align:top}
.plan th{font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); border-top:none}
.keep{color:#2e7d32; font-weight:700}
:root[data-theme="dark"] .keep{color:#7bd489}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .keep{color:#7bd489}}
.cut{color:#b3372c; font-weight:700}
:root[data-theme="dark"] .cut{color:#ff8a7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .cut{color:#ff8a7a}}
.pipe{display:flex; flex-wrap:wrap; gap:.4rem; align-items:center;
  font-size:.85rem; margin:.6rem 0 0}
.pipe .st{background:var(--amber-soft); color:var(--chip-ink);
  padding:.25rem .6rem; border-radius:5px; font-weight:600; white-space:nowrap}
.pipe .ar{color:var(--muted)}
code{font:.85em ui-monospace,Menlo,monospace; background:var(--card);
  border:1px solid var(--line); border-radius:4px; padding:.05em .35em}
.small{font-size:.85rem; color:var(--muted)}
"""

rows = [
    ("keep", "Teaser filmstrip (Fig. 1 style)", "It IS the pitch. Make it the full top-left column, bigger frames, one-line takeaway under it."),
    ("keep", "Method pipeline", "Rebuild from the 7 real-image panels below, left→right with arrows. No cartoon icons, at most one equation."),
    ("keep", "Recovery comparison (BundleSDF vs ours)", "Strongest evidence; keep the two-row filmstrip."),
    ("keep", "Main results", "One compact table (YCBMultiTrack-Real + HO3D means only) plus the ADD-S bar chart. Full tables → QR code."),
    ("keep", "YCBMultiTrack dataset card", "3 stats + one strip of dataset frames. Small."),
    ("cut", "“Key Idea” schematic with abstract track lines", "Replaced by the real trail panel (module 2) — same story, real pixels."),
    ("cut", "Equations for sampling / SDF refine / graph loss", "Poster readers won't parse them; keep at most the registration objective in small type."),
    ("cut", "Limitation scatter plots (texture analysis)", "Interesting in the paper, noise on a poster. One sentence in a footnote box if at all."),
    ("cut", "Component ablation table", "Compress to one line: “multi-hypothesis registration is the biggest contributor: ADD 65.5 → 82.8 on HO3D.”"),
]

row_html = "\n".join(
    f'<tr><td class="{k}">{"KEEP" if k=="keep" else "CUT"}</td><td>{what}</td><td>{how}</td></tr>'
    for k, what, how in rows
)

mods_html = ""
for m in MODULES:
    src, cap = m["primary"]
    alts = ""
    if m["alts"]:
        cells = "".join(
            f'<figure><img loading="lazy" src="{uri(a, 460, 78)}" alt="{c}">'
            f"<figcaption>{c}</figcaption></figure>"
            for a, c in m["alts"]
        )
        alts = f'<div class="alts">{cells}</div>'
    mods_html += f"""
<section class="mod">
  <div class="modhead"><span class="modnum">{m['num']}</span><h2>{m['title']}</h2></div>
  <p class="maps">replaces: {m['maps_to']}</p>
  <p class="note">{m['note']}</p>
  <figure><img src="{uri(src)}" alt="{cap}">
  <figcaption>{cap}<span class="pick">use this</span></figcaption></figure>
  {alts}
</section>"""

html = f"""<title>Point2Pose Poster — Module Figures</title>
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">ECCV 2026 · poster rebuild</div>
  <h1>Point2Pose poster: real-image module figures</h1>
  <p class="sub">Every panel below is rendered from actual run data — the saved
  <code>meta_data.npz</code> of the 3-object YCBMultiTrack-Real run
  (mustard&nbsp;/ SPAM&nbsp;/ tomato), plus reconstructed meshes and one HO3D run.
  Object colors are consistent everywhere: amber = mustard bottle,
  magenta = potted meat can, green = tomato soup can.
  Full-resolution files: <code>~/results/eccv_point2pose/paper_figs/poster_modules/</code></p>
</header>

<div class="plan">
  <h2>What changes vs. the current poster</h2>
  <table>
    <tr><th></th><th>Block</th><th>Recommendation</th></tr>
    {row_html}
  </table>
  <div class="pipe">
    <span class="st">① RGB-D + masks</span><span class="ar">→</span>
    <span class="st">② 2D point tracks</span><span class="ar">→</span>
    <span class="st">③ keypoint map</span><span class="ar">→</span>
    <span class="st">④ multi-hypothesis registration</span><span class="ar">→</span>
    <span class="st">⑤ pose graph</span><span class="ar">→</span>
    <span class="st">⑥ TSDF mesh</span><span class="ar">→</span>
    <span class="st">⑦ 6D poses</span>
  </div>
  <p class="small">Suggested pipeline strip for the poster middle column —
  each chip backed by the matching real-image panel below.</p>
</div>

{mods_html}

<section class="mod">
  <div class="modhead"><span class="modnum">↻</span><h2>Regenerating / retuning</h2></div>
  <p class="note">Scripts live in <code>scripts/poster_viz/</code> in the repo
  (run with the <code>ms</code> conda env). Each panel script takes a frame
  argument, e.g. <code>python panel_tracks.py ycb3 1290</code> or
  <code>python panel_tracks.py ho3d_AP14 auto</code>;
  <code>panel_hypo.py scan</code> lists the most ambiguous frames.
  Overlays render at 2× (1280×960); for print you may want to re-run with
  <code>S=3</code>.</p>
</section>
</main>
"""

with open(SHEET, "w") as f:
    f.write(html)
print("wrote", SHEET, f"{os.path.getsize(SHEET)/1e6:.1f} MB")
