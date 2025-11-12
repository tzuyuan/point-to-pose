function D = load_point2pose_run(base_path)
% LOAD_POINT2POSE_RUN  Load/unpack DataLogger output (.mat or .h5) into MATLAB-friendly structs.
%
% Usage:
%   D = load_point2pose_run('meta_data/meata_data');  % path WITHOUT extension
%
% Output struct D:
%   D.N                 - number of rows/frames (best-effort)
%   D.fixed             - struct of fixed-shape arrays (Nx..., numeric if consistent)
%   D.ragged            - struct of ragged fields as cell arrays, one cell per frame
%   D.helpers.regroup_pairs(C)   - reshape a cell array of 1-D vectors -> (M,2)
%   D.helpers.regroup_triples(C) - reshape a cell array of 1-D vectors -> (M,3)
%
% Notes:
% - Prefers .mat if present (fastest to use). Falls back to .h5 if needed.
% - Ragged sources handled in priority: *_cell (MAT) -> packed (*_data/_offsets/_lengths) -> vlen HDF5.

mat_path = [base_path '.mat'];
h5_path  = [base_path '.h5'];

has_mat = exist(mat_path, 'file') == 2;
has_h5  = exist(h5_path,  'file') == 2;

if ~has_mat && ~has_h5
    error('Neither %s nor %s found.', mat_path, h5_path);
end

% Known ragged fields from your logger config:
known_ragged = { ...
    'track2d','uncertainties','visibles','track3d','valid_depth', ...
    'obj_key_points','obj_uncertainties','obj_valid', 'obj_key_point_frames', ...
    'reg_key_points','reg_curr3d','reg_inliers','reg_residuals','reg_key_points_idx', ...
    'extract_vis_obj_mask', 'extract_val_obj_mask', 'extract_uncer_obj_mask',...
    'extract_valid_kp_mask','extract_uncertainty_thres','extract_obj_idx',...
    'extract_inside_mask','extract_finite_xy'
};

D = struct();
D.fixed  = struct();
D.ragged = struct();

if has_mat
    S = load(mat_path);   % loads into struct with fields
    % 1) Collect fixed fields (anything that is not packed/cell ragged)
    fields = fieldnames(S);
    for i = 1:numel(fields)
        k = fields{i};
        if endsWith(k,'_cell') || endsWith(k,'_data') || endsWith(k,'_offsets') || endsWith(k,'_lengths')
            continue; % handled as ragged below
        end
        % Likely fixed-shape numeric (or object-like cell already). Just copy.
        D.fixed.(k) = S.(k);
    end

    % 2) Ragged: prefer *_cell if present; else unpack packed form
    for i = 1:numel(known_ragged)
        name = known_ragged{i};
        cell_name = [name '_cell'];
        data_name = [name '_data'];  off_name = [name '_offsets'];  len_name = [name '_lengths'];

        if isfield(S, cell_name)
            D.ragged.(name) = S.(cell_name);
        elseif isfield(S, data_name) && isfield(S, off_name) && isfield(S, len_name)
            D.ragged.(name) = unpackPackedFromStruct(name, S);
        elseif isfield(S, name)
            % Sometimes ragged might have been saved directly (rare). Wrap as cell row-wise.
            arr = S.(name);
            if iscell(arr), D.ragged.(name) = arr;
            else,           D.ragged.(name) = num2cell(arr, 2); % default split rows
            end
        end
    end

elseif has_h5
    % ---------- HDF5 path ----------
    info = h5info(h5_path);
    names = {info.Datasets.Name};

    % Helper to test dataset presence
    has = @(k) any(strcmp(names,k));

    % Fixed: read everything that's not in known ragged (we can't perfectly infer shapes here)
    for i = 1:numel(names)
        k = names{i};
        if ismember(k, known_ragged)
            continue;
        end
        D.fixed.(k) = h5read(h5_path, ['/' k]);
    end

    % Ragged: HDF5 vlen datasets read as cell arrays in MATLAB
    for i = 1:numel(known_ragged)
        name = known_ragged{i};
        if has(name)
            D.ragged.(name) = h5read(h5_path, ['/' name]);
        elseif has([name '_data']) && has([name '_offsets']) && has([name '_lengths'])
            S_h5 = struct();
            S_h5.([name '_data'])    = h5read(h5_path, ['/' name '_data']);
            S_h5.([name '_offsets']) = h5read(h5_path, ['/' name '_offsets']);
            S_h5.([name '_lengths']) = h5read(h5_path, ['/' name '_lengths']);
            D.ragged.(name) = unpackPackedFromStruct(name, S_h5);
        end
    end
end

% ---- Determine N (rows/frames) best-effort ----
candidates = {};
if isfield(D.fixed,'frame_id'), candidates{end+1} = numel(D.fixed.frame_id); end
if isfield(D.fixed,'timestamp'), candidates{end+1} = numel(D.fixed.timestamp); end
% If any ragged exists, use its length as N
rag_names = fieldnames(D.ragged);
for i = 1:numel(rag_names)
    if iscell(D.ragged.(rag_names{i}))
        candidates{end+1} = numel(D.ragged.(rag_names{i}));
    end
end
if isempty(candidates)
    D.N = NaN;
else
    D.N = max([candidates{:}]);
end

% ---- helpers to regroup flattened ragged vectors into (M,2) or (M,3) ----
D.helpers.regroup_pairs = @(C) cellfun(@(v) reshapeSafe(v, 2), C, 'UniformOutput', false);
D.helpers.regroup_triples = @(C) cellfun(@(v) reshapeSafe(v, 3), C, 'UniformOutput', false);

fprintf('[load_point2pose_run] Loaded. N ~= %d rows. Fixed=%d fields, Ragged=%d fields.\n', ...
        D.N, numel(fieldnames(D.fixed)), numel(fieldnames(D.ragged)));

end % function


% ================== local helpers ==================

function C = unpackPackedFromStruct(name, S)
% Rebuild a cell array from packed triplet: <name>_data, _offsets, _lengths
data = S.([name '_data']);
offs = S.([name '_offsets']);
lens = S.([name '_lengths']);
N = numel(lens);
C = cell(N,1);
for i = 1:N
    if lens(i) == 0
        C{i} = data([]); %#ok<NASGU>
        C{i} = zeros(0,1, class(data)); % empty of right dtype
    else
        idx = (offs(i)+1):(offs(i)+lens(i)); % MATLAB 1-based
        C{i} = data(idx);
    end
end
end

function A = reshapeSafe(v, k)
% Reshape 1-D vector v into (#rows,k). If length not divisible by k, return v as column.
if isempty(v)
    A = reshape(v, 0, k);
    return;
end
n = numel(v);
if mod(n, k) ~= 0
    A = v(:); % can't reshape cleanly
else
    A = reshape(v, k, []).';
end
end