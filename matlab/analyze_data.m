D = load_point2pose_run('debug/pipeline/meta_data/meata_data');

% --- Fixed fields ---
frame_id  = getfield(D.fixed, 'frame_id');     %#ok<GFLD>
timestamp = getfield(D.fixed, 'timestamp');    %#ok<GFLD>

% --- Ragged examples ---
residuals = D.ragged.reg_residuals;   % cell {N x 1}, each is Mi x 1 double
inliers   = D.ragged.reg_inliers;     % cell {N x 1}, each is Mi x 1 logical

% Compute per-frame stats
N = D.N;
inlier_ratio = nan(N,1);
mean_inl_res = nan(N,1);
for i = 1:N
    if i <= numel(residuals) && i <= numel(inliers)
        r = residuals{i};  il = inliers{i};
        if ~isempty(r)
            inlier_ratio(i) = mean(il);
            if any(il), mean_inl_res(i) = mean(r(il)); end
        end
    end
end

% --- Rebuild 2D arrays from flattened ragged data ---
% track2d was saved as flattened 1-D vectors; reshape into (M,2) per frame:
if isfield(D.ragged, 'track2d')
    track2d_xy = D.helpers.regroup_pairs(D.ragged.track2d);  % each cell: (#points, 2)
end

% track3d -> reshape into (M,3) per frame:
if isfield(D.ragged, 'track3d')
    track3d_xyz = D.helpers.regroup_triples(D.ragged.track3d); % each cell: (#points, 3)
end

% --- Poses: if saved as fixed (N x 4 x 4) ---
if isfield(D.fixed, 'obj_pose')
    poses = D.fixed.obj_pose;   % expected (4,4,N) or (N,4,4) depending on writer
    sz = size(poses);
    if numel(sz) == 3 && sz(1)==4 && sz(2)==4
        % shape (4,4,N)
        T = poses;  % use as-is
    elseif numel(sz) == 3 && sz(2)==4 && sz(3)==4
        % shape (N,4,4) -> permute to (4,4,N)
        T = permute(poses, [2 3 1]);
    else
        % Fallback: if object array of matrices
        if iscell(poses)
            Np = numel(poses);
            T = nan(4,4,Np);
            for i = 1:Np
                if ~isempty(poses{i}), T(:,:,i) = poses{i}; end
            end
        else
            T = [];  % unknown layout
        end
    end
    % Example: extract translation (tx,ty,tz) per frame
    if ~isempty(T)
        txyz = squeeze(T(1:3,4,:)).';
    end
end