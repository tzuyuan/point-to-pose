clear;
clc;

D = load_point2pose_run('debug/pipeline/meta_data/meata_data');

% --- Fixed fields ---
frame_id  = getfield(D.fixed, 'frame_id');     %#ok<GFLD>
timestamp = getfield(D.fixed, 'timestamp');    %#ok<GFLD>


i = 209;

reg_key_points =  D.helpers.regroup_triples(D.ragged.reg_key_points);
cur3d =  D.helpers.regroup_triples(D.ragged.reg_curr3d);
track3d = D.helpers.regroup_triples(D.ragged.track3d);

uncertainty_i = D.ragged.uncertainties{i};
reg_key_points_idx_i = D.ragged.reg_key_points_idx{i};
reg_uncertainty = uncertainty_i(reg_key_points_idx_i);
reg_key_points_i = reg_key_points{i};
cur3d_i = cur3d{i};
track3d_i = track3d{i};

% --- Find indices of curr3d_i in track3d_i ---
tol = 1e-6;  % tolerance for floating-point comparison

idx_in_track = zeros(size(cur3d_i,1),1);  % preallocate

for k = 1:size(cur3d_i,1)
    diffs = vecnorm(track3d_i - cur3d_i(k,:), 2, 2);
    [min_val, min_idx] = min(diffs);
    if min_val < tol
        idx_in_track(k) = min_idx;
    else
        idx_in_track(k) = NaN; % no match found
    end
end

% Remove NaN if you only want valid matches
valid_idx = idx_in_track(~isnan(idx_in_track));



visible_i = D.ragged.visibles{i};
uncertainties_i = D.ragged.uncertainties{i};
inlier_i = D.ragged.reg_inliers{i};
residuals_i = D.ragged.reg_residuals{i};
residuals_inlier_i = residuals_i(inlier_i);
world_pose_i = reshape(D.fixed.obj_pose(i,:,:),4,4);
init_pose = reshape(D.fixed.obj_init_pose(i,:,:),4,4);
pose_i = world_pose_i;

track2d_xy = D.helpers.regroup_pairs(D.ragged.track2d);

apply_pose = true;       % set false if reg_key_points_i is already in the same frame as cur3d_i
show_lines = true;
src_marker = 'o';
tgt_marker = 's';
base_size  = 24;         % base marker size
unc_range  = [prctile(uncertainties_i, 10), prctile(uncertainties_i, 90)]; % robust range for sizing
unc_range(isnan(unc_range)) = 1; % fallback if uncertainties_i missing
line_alpha = 0.6;        % transparency of lines (requires R2022a+ for RGBA in plot3; else uses solid)
line_width = 1.5;

% ------------------- validate & prep -------------------
assert(size(reg_key_points_i,2)==3 && size(cur3d_i,2)==3, 'Points must be N×3.');
N = min(size(reg_key_points_i,1), size(cur3d_i,1));
reg_key_points_i = reg_key_points_i(1:N, :);
cur3d_i          = cur3d_i(1:N, :);

% Visibility mask (optional)
% if exist('visible_i','var') && ~isempty(visible_i)
%     mask = logical(visible_i(1:N));
% else
mask = true(N,1);
% end

% Apply pose to source (optional)
if apply_pose && exist('pose_i','var') && ~isempty(pose_i) && all(size(pose_i)==[4 4])
    R = pose_i(1:3,1:3);
    t = pose_i(1:3,4).';
    reg_src = (R * reg_key_points_i.' + t.').';
else
    reg_src = reg_key_points_i;
end

% Filter by mask
reg_src = reg_src(mask, :);
cur3d   = cur3d_i(mask, :);
M = size(reg_src,1);

% ------------------- colors & sizes -------------------
% Same color per pair
if M<=0
    warning('No points to plot after masking.'); 
    figure; axis off; title('No correspondences to display'); 
    return;
end
C = hsv(max(M,8));          % vivid, distinct; ensures at least 8 colors
C = C(1:M, :);

% Marker sizes from uncertainties (optional)
if exist('uncertainties_i','var') && ~isempty(uncertainties_i)
    u = uncertainties_i(1:N);
    u = u(mask);
    if any(isfinite(u))
        u = max(unc_range(1), min(unc_range(2), u)); % clamp
        u = (u - unc_range(1)) / max(eps, (unc_range(2)-unc_range(1)));
        sz_src = base_size * (0.6 + 1.4*u);   % 0.6x .. 2.0x
        sz_tgt = base_size * (0.6 + 1.4*u);
    else
        sz_src = base_size * ones(M,1);
        sz_tgt = base_size * ones(M,1);
    end
else
    sz_src = base_size * ones(M,1);
    sz_tgt = base_size * ones(M,1);
end

% ------------------- plot -------------------
figure('Color','w'); hold on; grid on; axis equal
xlabel('X'); ylabel('Y'); zlabel('Z'); view(45,25);

% Draw correspondence lines efficiently using NaN-separated segments
if show_lines
    X = [reg_src(:,1), cur3d(:,1), nan(M,1)]';
    Y = [reg_src(:,2), cur3d(:,2), nan(M,1)]';
    Z = [reg_src(:,3), cur3d(:,3), nan(M,1)]';
    % Plot all segments in one call; color per segment by looping colors once (fast enough for typical M)
    % (MATLAB's plot3 can't assign per-segment colors in a single call, so we do a lightweight loop.)
    for k = 1:M
        pk = plot3(X(1:2,k), Y(1:2,k), Z(1:2,k), '-', 'LineWidth', line_width);
        pk.Color = [C(k,:), line_alpha];  % uses RGBA if supported; otherwise ignores alpha
    end
end

% Plot source and target points with same per-pair colors
hs = scatter3(reg_src(:,1), reg_src(:,2), reg_src(:,3), sz_src, C, src_marker, 'filled', 'MarkerFaceAlpha', 0.95);
ht = scatter3(cur3d(:,1),   cur3d(:,2),   cur3d(:,3),   sz_tgt, C, tgt_marker, 'filled', 'MarkerFaceAlpha', 0.95);

legend([hs, ht], {'Source (transformed)', 'Target'}, 'Location','best');
title(sprintf('3D Correspondences (N=%d) — frame %d', M, double(frame_id(i))));

% Nice bounds
all_pts = [reg_src; cur3d];
mins = min(all_pts,[],1); maxs = max(all_pts,[],1);
rng_pad = 0.05*(maxs - mins + eps);
xlim([mins(1)-rng_pad(1), maxs(1)+rng_pad(1)]);
ylim([mins(2)-rng_pad(2), maxs(2)+rng_pad(2)]);
zlim([mins(3)-rng_pad(3), maxs(3)+rng_pad(3)]);