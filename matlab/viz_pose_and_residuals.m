function viz_pose_and_residuals(pose_dir, rad_units)
% Visualize poses and registration stats vs TIME, with frame-id on a top x-axis.
%
% Stats format (13 cols):
% timestamp frame_id obj_id num_points iter thr res_mean res_median res_max ...
% num_inliers total_points mean_residual_inliers mean_residual_outliers
%
% Usage:
%   viz_pose_and_residuals_time("./poses");        % radians (default)
%   viz_pose_and_residuals_time("./poses", false); % degrees

if nargin < 1 || isempty(pose_dir), pose_dir = "./poses"; end
if nargin < 2, rad_units = true; end

% ---- discover pose files per object ----
pose_files = dir(fullfile(pose_dir, "obj_*_pose.txt"));
alt_pose_files = dir(fullfile(pose_dir, "pose_rpy_xyz*.txt")); % optional alt format
if isempty(pose_files) && isempty(alt_pose_files)
    error("No pose logs found in %s", pose_dir);
end

% ---- read registration stats (with time) ----
stats_path = fullfile(pose_dir, "registration_stats.txt");
has_stats = isfile(stats_path);
if has_stats
    stats = read_registration_stats_with_time_v2(stats_path);
    obj_ids_in_stats = unique(stats.obj_id);
else
    stats = []; obj_ids_in_stats = [];
end

% ---- collect object ids from pose filenames ----
obj_ids_from_pose = [];
for k = 1:numel(pose_files)
    m = regexp(pose_files(k).name, "obj_(\d+)_pose\.txt", "tokens", "once");
    if ~isempty(m), obj_ids_from_pose(end+1) = str2double(m{1}); end %#ok<AGROW>
end
obj_ids_from_pose = unique(obj_ids_from_pose);
if isempty(obj_ids_from_pose) && ~isempty(alt_pose_files)
    obj_ids_from_pose = 0; % fallback single object
end
all_obj_ids = unique([obj_ids_from_pose, obj_ids_in_stats]);

for oid = all_obj_ids
    % ---- load pose for this object (TUM or fallback to alt format) ----
    if any(obj_ids_from_pose == oid)
        fpath = fullfile(pose_dir, sprintf("obj_%d_pose.txt", oid));
        pose = read_tum_pose_file(fpath);
    else
        fpath = fullfile(pose_dir, alt_pose_files(1).name);
        pose = read_rpy_xyz_pose_file(fpath);
    end
    if isempty(pose.t)
        warning("No pose rows for object %d", oid);
        continue;
    end

    % quat -> rpy (radians)
    if isfield(pose, "quat"), rpy = quat_to_rpy(pose.quat); else, rpy = pose.rpy; end
    ang = rpy; if ~rad_units, ang = rad2deg(ang); end

    % pick stats rows for this object
    if has_stats
        mask = stats.obj_id == oid;
        s = subset_stats(stats, mask);
    else
        s = [];
    end

    % align time origins
    if has_stats && ~isempty(s.timestamp)
        t0 = min(pose.t(1), s.timestamp(1));
        tt_pose  = pose.t - t0;
        tt_stats = s.timestamp - t0;
    else
        t0 = pose.t(1);
        tt_pose  = pose.t - t0;
        tt_stats = [];
    end

    % ================= Pose figure =================
    fig1 = figure('Name', sprintf('Object %d — Pose (time)', oid), 'Color', 'w');
    tl = tiledlayout(fig1, 3, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf('Object %d: Pose vs. time', oid));

    nexttile; plot(tt_pose, ang(:,1), 'LineWidth',1.2); grid on
    ylabel(lbl('Roll', rad_units)); xlabel('time [s]'); title('Roll')

    nexttile; plot(tt_pose, ang(:,2), 'LineWidth',1.2); grid on
    ylabel(lbl('Pitch', rad_units)); xlabel('time [s]'); title('Pitch')

    nexttile; plot(tt_pose, ang(:,3), 'LineWidth',1.2); grid on
    ylabel(lbl('Yaw', rad_units)); xlabel('time [s]'); title('Yaw')

    nexttile; plot(tt_pose, pose.xyz(:,1), 'LineWidth',1.2); grid on
    ylabel('x [m]'); xlabel('time [s]'); title('x')

    nexttile; plot(tt_pose, pose.xyz(:,2), 'LineWidth',1.2); grid on
    ylabel('y [m]'); xlabel('time [s]'); title('y')

    nexttile; plot(tt_pose, pose.xyz(:,3), 'LineWidth',1.2); grid on
    ylabel('z [m]'); xlabel('time [s]'); title('z')

    % ================= Residuals figure (time + top frame-id) =================
    if has_stats && ~isempty(s.timestamp)
        fig2 = figure('Name', sprintf('Object %d — Registration (time)', oid), 'Color', 'w');
        tl2 = tiledlayout(fig2, 2, 2, 'TileSpacing','compact', 'Padding','compact');
        title(tl2, sprintf('Object %d: Registration stats vs. time', oid));

        step = max(1, floor(numel(s.frame_id)/8));
        xticks_idx = 1:step:numel(s.frame_id);
        addTopAxis = @(ax) add_top_frame_axis(ax, tt_stats, s.frame_id, xticks_idx);

        % residuals
        ax = nexttile; hold on; grid on
        plot(tt_stats, s.res_mean,   'LineWidth',1.4);
        plot(tt_stats, s.res_median, 'LineWidth',1.2, 'LineStyle','--');
        plot(tt_stats, s.res_max,    'LineWidth',1.2, 'LineStyle',':');
        ylabel('residual [m]'); xlabel('time [s]');
        legend({'mean','median','max'}, 'Location','best'); title('Residuals');
        addTopAxis(ax);

        % threshold
        ax = nexttile; plot(tt_stats, s.thr, 'LineWidth',1.2); grid on
        ylabel('threshold [m]'); xlabel('time [s]'); title('Threshold');
        addTopAxis(ax);

        % inliers
        ax = nexttile; plot(tt_stats, s.num_inliers, 'LineWidth',1.2); grid on
        ylabel('# inliers'); xlabel('time [s]'); title('Inliers');
        addTopAxis(ax);

        % points used
        ax = nexttile; plot(tt_stats, s.num_points, 'LineWidth',1.2); grid on
        ylabel('# points used'); xlabel('time [s]'); title('Points used');
        addTopAxis(ax);

        linkaxes(findall(fig2,'type','axes'),'x');

        % Residuals figure datatip with frame id
        dcm2 = datacursormode(fig2);
        dcm2.Enable = 'on';
        dcm2.UpdateFcn = @(~,evt) cursorTextWithFrame(evt, tt_stats, s.frame_id);

        % ================= Combined: mean residual + R/P/Y =================
        fig3 = figure('Name', sprintf('Object %d — Mean residual & RPY (time)', oid), 'Color','w');
        axc = axes(fig3); hold(axc,'on'); grid(axc,'on')
        yyaxis(axc,'left');  plot(tt_stats, s.res_mean, 'LineWidth',1.6);
        ylabel(axc,'mean residual [m]');
        yyaxis(axc,'right'); plot(tt_pose,  ang(:,1), '-',  'LineWidth',1.1);
                             plot(tt_pose,  ang(:,2), '--', 'LineWidth',1.1);
                             plot(tt_pose,  ang(:,3), ':',  'LineWidth',1.1);
        ylabel(axc, sprintf('angle [%s]', ternary(rad_units,'rad','deg')));
        xlabel(axc,'time [s]');
        title(axc, sprintf('Object %d: Mean residual (left) vs Roll/Pitch/Yaw (right)', oid));
        legend({'mean residual','roll','pitch','yaw'}, 'Location','best');
        add_top_frame_axis(axc, tt_stats, s.frame_id, xticks_idx);
        dcm3 = datacursormode(fig3); dcm3.Enable = 'on';
        dcm3.UpdateFcn = @(~,evt) cursorTextWithFrame(evt, tt_stats, s.frame_id);

        % ================= Angle combo plots: angle + mean_residual_inliers + inlier_ratio =================
        denom = max(1, s.total_points);                   % guard divide-by-zero
        inlier_ratio = s.num_inliers ./ denom;            % 0..1 (or NaN if inputs NaN)
        mri = s.mean_residual_inliers;                    % meters

        make_angle_combo_plot('Roll',  ang(:,1), tt_pose, tt_stats, inlier_ratio, mri, rad_units, oid, s.frame_id);
        make_angle_combo_plot('Pitch', ang(:,2), tt_pose, tt_stats, inlier_ratio, mri, rad_units, oid, s.frame_id);
        make_angle_combo_plot('Yaw',   ang(:,3), tt_pose, tt_stats, inlier_ratio, mri, rad_units, oid, s.frame_id);
    end
end
end

% ===================== Helpers =====================

function s = read_tum_pose_file(fpath)
raw = readtext_numeric_robust(fpath, '#');
if isempty(raw) || size(raw,2) < 8
    s = struct('t',[],'xyz',[],'quat',[]);
    return;
end
s.t    = raw(:,1);
s.xyz  = raw(:,2:4);
s.quat = raw(:,5:8); % [qx qy qz qw]
end

function s = read_rpy_xyz_pose_file(fpath)
raw = readtext_numeric_robust(fpath, '#');
if isempty(raw) || size(raw,2) < 7
    s = struct('t',[],'rpy',[],'xyz',[]);
    return;
end
s.t   = raw(:,1);
s.rpy = raw(:,2:4);
s.xyz = raw(:,5:7);
end

function stats = read_registration_stats_with_time_v2(fpath)
% 13-column format:
% timestamp frame_id obj_id num_points iter thr res_mean res_median res_max ...
% num_inliers total_points mean_residual_inliers mean_residual_outliers
T = readtable(fpath, 'FileType','text', 'Delimiter','\t', 'ReadVariableNames',false);
if width(T) ~= 13
    error("Unexpected format in %s (expected 13 columns, got %d).", fpath, width(T));
end
stats = table2struct(T, 'ToScalar', true);
stats.timestamp               = T.Var1;
stats.frame_id                = T.Var2;
stats.obj_id                  = T.Var3;
stats.num_points              = T.Var4;
stats.iter                    = T.Var5;
stats.thr                     = T.Var6;
stats.res_mean                = T.Var7;
stats.res_median              = T.Var8;
stats.res_max                 = T.Var9;
stats.num_inliers             = T.Var10;
stats.total_points            = T.Var11;
stats.mean_residual_inliers   = T.Var12;
stats.mean_residual_outliers  = T.Var13;
end

function out = subset_stats(s, mask)
fn = fieldnames(s);
for i = 1:numel(fn)
    v = s.(fn{i});
    out.(fn{i}) = v(mask);
end
end

function data = readtext_numeric_robust(fpath, comment_char)
% Reads whitespace-separated numeric file, skipping empty/comment lines.
fid = fopen(fpath,'r'); if fid < 0, error("Cannot open %s", fpath); end
C = {};
while true
    L = fgetl(fid); if ~ischar(L), break; end
    L = strtrim(L);
    if isempty(L), continue; end
    if nargin >= 2 && ~isempty(comment_char) && startsWith(L, comment_char), continue; end
    nums = sscanf(L, '%f');
    if ~isempty(nums), C{end+1,1} = nums(:).'; end %#ok<AGROW>
end
fclose(fid);
if isempty(C), data = [];
else
    maxlen = max(cellfun(@numel, C));
    data = nan(numel(C), maxlen);
    for i = 1:numel(C), v = C{i}; data(i,1:numel(v)) = v; end
end
end

function rpy = quat_to_rpy(q)
% Quaternion [qx qy qz qw] -> [roll pitch yaw] (XYZ intrinsic / ZYX extrinsic)
qx = q(:,1); qy = q(:,2); qz = q(:,3); qw = q(:,4);
R11 = 1 - 2*(qy.^2 + qz.^2);
R21 = 2*(qx.*qy + qz.*qw);
R31 = 2*(qx.*qz - qy.*qw);
R32 = 2*(qy.*qz + qx.*qw);
R33 = 1 - 2*(qx.^2 + qy.^2);
pitch = atan2(-R31, sqrt(max(0, 1 - R31.^2)));
roll  = atan2(R32, R33);
yaw   = atan2(R21, R11);
rpy = [roll, pitch, yaw];
end

function s = lbl(name, rad_units)
if rad_units, s = sprintf('%s [rad]', name);
else,         s = sprintf('%s [deg]', name); end
end

function s = ternary(cond, a, b)
if cond, s = a; else, s = b; end
end

function add_top_frame_axis(ax, tt_stats, frame_ids, xticks_idx)
% Overlay a top x-axis with frame IDs aligned to time.
if isempty(tt_stats) || isempty(frame_ids), return; end
fig = ancestor(ax,'figure');
ax2 = axes('Parent', fig, ...
           'Position', ax.Position, ...
           'XAxisLocation', 'top', ...
           'YAxisLocation', 'right', ...
           'Color', 'none', ...
           'XColor', [0.3 0.3 0.3], ...
           'YColor', 'none', ...
           'HitTest','off', 'PickableParts','none');   % non-interactive
ax2.XLim = ax.XLim;
set(ax2, 'XTick', tt_stats(xticks_idx), ...
         'XTickLabel', string(frame_ids(xticks_idx)), ...
         'TickDir', 'out', 'FontSize', 8);
xlabel(ax2, 'frame id');
uistack(ax2,'top');  % draw top axis on top so labels are visible
uistack(ax,'top');   % keep main axes interactive for clicks
end

function make_angle_combo_plot(angle_name, angle_series, tt_pose, tt_stats, inlier_ratio, mri, rad_units, oid, frame_ids)
% One figure per angle:
%   Left  y-axis: angle (roll/pitch/yaw)
%   Right y-axis: mean_residual_inliers [m]  AND  inlier_ratio (scaled into residual range)
%   Bottom x-axis: time [s]; Top x-axis: frame id
%   Datatips show time + nearest frame id

fig = figure('Name', sprintf('Object %d — %s + residual(inliers) + inlier ratio', oid, angle_name), ...
             'Color','w');
ax = axes('Parent', fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% --- Sanity/shape guards
tt_pose  = tt_pose(:);
tt_stats = tt_stats(:);
angle_series = angle_series(:);
mri = mri(:);
if isempty(inlier_ratio), inlier_ratio = nan(size(tt_stats)); end
inlier_ratio = inlier_ratio(:);

% --- Left Y: angle
yyaxis(ax,'left');
pA = plot(ax, tt_pose, angle_series, 'LineWidth', 1.8);
ylabel(ax, sprintf('%s [%s]', angle_name, ternary(rad_units,'rad','deg')));

% --- Right Y: residual (meters) + scaled inlier_ratio
yyaxis(ax,'right');

% Robust residual range
valid_mri = isfinite(mri);
if any(valid_mri)
    rmin = min(mri(valid_mri));
    rmax = max(mri(valid_mri));
else
    rmin = 0; rmax = 1;
end
if rmin == rmax
    pad = max(1e-3, 0.1*max(1,abs(rmax)));
    rmin = rmin - pad; rmax = rmax + pad;
end

% Scale inlier_ratio (0..1) into [rmin, rmax]
valid_ir = isfinite(inlier_ratio);
ir_scaled = nan(size(inlier_ratio));
ir_scaled(valid_ir) = rmin + inlier_ratio(valid_ir) * (rmax - rmin);

pR  = plot(ax, tt_stats, mri,       'LineWidth', 1.8, 'LineStyle', '--');      % residual (meters)
pIR = plot(ax, tt_stats, ir_scaled, 'LineWidth', 1.8, 'LineStyle', ':', ...
                         'Marker', 'o', 'MarkerSize', 4, ...
                         'MarkerIndices', 1:max(1,floor(numel(ir_scaled)/20)):numel(ir_scaled),'Color', [0.00 0.60 0.20]);
ylabel(ax, 'mean residual (inliers) [m]');

% Annotate scale hint for the ratio so you remember it's scaled
txt_hint = sprintf('inlier ratio scaled: 0\\rightarrow%.3g, 1\\rightarrow%.3g', rmin, rmax);
text(ax, double(ax.XLim(1)), double(rmax), ['  ' txt_hint], ...
     'VerticalAlignment','bottom', 'FontSize',8, 'Color',[0.2 0.5 0.2]);

% --- Top x-axis: frame id (on main axes)
if ~isempty(tt_stats)
    step = max(1, floor(numel(frame_ids)/8));
    xticks_idx = 1:step:numel(frame_ids);
    add_top_frame_axis(ax, tt_stats, frame_ids, xticks_idx);
end

% Labels / legend / title
xlabel(ax, 'time [s]');
title(ax, sprintf('Object %d: %s + mean residual (inliers) + inlier ratio', oid, angle_name));
legend(ax, [pA, pR, pIR], {angle_name, 'mean residual (inliers) [m]', 'inlier ratio (scaled to right axis)'}, ...
       'Location','best');

% --- Datatips with nearest frame id
if ~isempty(tt_stats)
    dcm = datacursormode(fig);
    dcm.Enable = 'on';
    dcm.UpdateFcn = @(~,evt) cursorTextWithFrame(evt, tt_stats, frame_ids);
end
end

function txt = cursorTextWithFrame(evt, tt_stats, frame_ids)
t = evt.Position(1);
y = evt.Position(2);
[~, k] = min(abs(tt_stats - t));
fr = NaN; if ~isempty(k), fr = frame_ids(k); end
txt = sprintf('time: %.3f s\nframe: %d\nvalue: %.6g', t, fr, y);
end