function run_dir = run_calado_driftfusion_reference(source_json, options)
% Independent Driftfusion v1.1.1 comparison of a frozen SolarLab Calado run.
% Downloads the official, checksum-pinned archive into ignored outputs.
% Keeps original upstream code and SolarLab kernels intact; observes a copy.
% This is a common-input solver comparison, not the original 2016 source deck.
% File checks use native MATLAB paths and /usr/bin/openssl (macOS/Linux).
arguments
    source_json (1,1) string = ""
    options.LayerPoints (1,3) double {mustBeInteger,mustBePositive} = [300 600 300]
    options.RelTol (1,1) double {mustBePositive} = 1e-4
    options.AbsTol_cm3 (1,1) double {mustBePositive} = 1e-6
    options.MajorityVelocity_cm_s (1,1) double {mustBePositive} = 1e8
    options.ScanPoints (1,1) double {mustBeInteger,mustBePositive} = 441
    options.StageTimeout_s (1,1) {mustBeNumeric,mustBeReal,mustBeFinite,mustBePositive} = 30
    options.RunTimeout_s (1,1) {mustBeNumeric,mustBeReal,mustBeFinite,mustBePositive} = 300
    options.NeutralContactIonPadding (1,1) {mustBeA(options.NeutralContactIonPadding, 'logical')} = false
    options.UniformNodes (1,1) {mustBeNumeric,mustBeReal,mustBeFinite,mustBeInteger,mustBeNonnegative} = 0
end
run_started = tic;
project = fileparts(fileparts(mfilename('fullpath')));
if source_json == ""
    source_json = fullfile(project, 'outputs', 'calado-reproduction', ...
        'alignment', 'paper-contacts-atol1-n60', 'alignment.json');
end
assert(isfile(source_json), 'Source JSON file does not exist: %s', source_json);
[path_ok, source_info] = fileattrib(source_json);
assert(path_ok && isscalar(source_info) && ~source_info.directory, ...
    'Cannot resolve source JSON file: %s', source_json);
source_json = string(source_info.Name);
assert(options.ScanPoints >= 3, 'At least three scan points are required.');
assert(options.UniformNodes == 0 || options.UniformNodes >= 4, 'UniformNodes must be zero or at least four.');
source = jsondecode(fileread(source_json));
assert(strcmp(source.status, 'completed'), 'Source run must be completed.');
assert(~source.settings.no_contact_srh, 'Use the contact-SRH source run.');
assert(source.settings.voltage_ramp, 'Source must use continuous voltage ramps.');
assert(source.settings.prepare_ramp_seconds == 0, 'Dark bias ramp is not implemented here.');
scan_start = -1;
scan_end = 1.2;
if isfield(source.settings, 'scan_start_voltage')
    scan_start = source.settings.scan_start_voltage;
end
if isfield(source.settings, 'scan_end_voltage')
    scan_end = source.settings.scan_end_voltage;
end
assert(isnumeric(scan_start) && isscalar(scan_start) && isreal(scan_start) && isfinite(scan_start), ...
    'Source scan start must be a finite real numeric scalar.');
assert(isnumeric(scan_end) && isscalar(scan_end) && isreal(scan_end) && isfinite(scan_end) && scan_end > max(0, scan_start), ...
    'Source scan end must be finite, positive, and above scan start.');
assert(source.settings.prepare_voltage == -1, 'This adapter requires dark preparation at -1 V.');
scan_time = (scan_end - scan_start) / source.settings.scan_rate;
assert(isfinite(scan_time) && scan_time > 0, 'Source must define a positive scan duration.');
stack = source.device_stack;
assert(numel(stack.layers) == 3 && strcmp(stack.mode, 'full'), 'Expected the three-layer Full-mode Calado probe.');
assert(stack.T == 300, 'This common-input adapter is restricted to 300 K.');
assert(stack.ion_steric_diffusion_only, 'Expected the diffusion-only finite-site ion model.');
assert(stack.S_n_left == 0 && stack.S_p_right == 0, 'Expected blocking minority contacts.');
assert(isempty(stack.S_p_left) && isempty(stack.S_n_right), 'Expected fixed-majority SolarLab contacts.');
assert(all(stack.interfaces == 0, 'all'), 'Interface recombination must be zero.');

out_root = fullfile(project, 'outputs', 'calado-reproduction', 'driftfusion');
if ~isfolder(out_root), mkdir(out_root); end
upstream = obtain_release(out_root);
run_dir = fullfile(out_root, ['run-' char(datetime('now','Format','yyyyMMdd-HHmmss-SSS'))]);
assert(~isfolder(run_dir), 'Run directory already exists.');
mkdir(run_dir);
old_path = path;
old_dir = pwd;
environment_cleanup = onCleanup(@() restore_environment(old_path, old_dir));
addpath(genpath(upstream));
cd(upstream);
assert(startsWith(which('df'), upstream), 'Unexpected df function on the MATLAB path.');
monitored_driver = prepare_calado_monitored_driver(which('df'), run_dir);
addpath(run_dir);
assert(strcmp(which('df_calado_monitored'), monitored_driver), 'Unexpected monitored driver on path.');

metadata = struct('status', 'running', 'MATLAB_version', version, ...
    'upstream_release', 'v1.1.1', 'upstream_archive_md5', 'b8d2fd78d9467967af94c4f617ffea92', ...
    'upstream_doi', '10.5281/zenodo.6045592', 'source_json', char(source_json), ...
    'source_json_sha256', file_digest(source_json, 'SHA-256'), ...
    'adapter_sha256', file_digest([mfilename('fullpath') '.m'], 'SHA-256'), ...
    'upstream_df_sha256', file_digest(which('df'), 'SHA-256'), ...
    'upstream_pc_sha256', file_digest(which('pc'), 'SHA-256'), ...
    'monitored_driver_sha256', file_digest(monitored_driver, 'SHA-256'), ...
    'monitor_note', ['Derived df copy adds only one observation hook per PDE mesh evaluation; ' ...
        'no equation or ode15s/pdepe option changes. Limits are cooperative, not OS timeouts.'], ...
    'bias_initialization_note', ['At a bias step, add a homogeneous Poisson lift to the previous ' ...
        'potential. Density states and physical history durations remain unchanged.'], ...
    'mesh_note', ['UniformNodes=0 uses interface-aligned layer meshes; otherwise a single ' ...
        'global linspace with exactly UniformNodes nodes is compiled.'], ...
    'options', options, 'source', source, ...
    'scan_protocol', struct('start_voltage_V', scan_start, 'end_voltage_V', scan_end, 'branch_duration_s', scan_time), ...
    'scope', 'Common-input external solver check; not the original 2016 deck', ...
    'contact_note', 'Minorities blocked; finite majority S approximates Dirichlet. Refine S independently.', ...
    'steric_note', 'Upstream finite-site diffusion retained at the exported dilute P/P_lim.', ...
    'contact_ion_note', ['NeutralContactIonPadding is explicit: when enabled, contacts carry ' ...
        'equal cation state and static background as in legacy commit c46f54b; mobility remains zero. ' ...
        'No SolarLab input or production equation is changed.'], ...
    'initialization_note', ['Upstream-equilibrate ordering: the first electrostatic/electronic ' ...
        'stages use three variables, then the configured cation state is appended analytically ' ...
        'before the exported dark-0 history. Radiation and SRH are disabled only during ' ...
        'the 1e-12 s zero-transport seed and restored for electronic relaxation and all history.']);
helper_names = {'prepare_calado_monitored_driver', 'calado_reference_monitor', ...
    'calado_reference_bias_lift', 'validate_calado_reference_solution'};
for helper_index = 1:numel(helper_names)
    helper_name = helper_names{helper_index};
    helper_file = fullfile(project, 'scripts', [helper_name '.m']);
    assert(strcmp(which(helper_name), helper_file), 'Unexpected reference helper on MATLAB path.');
    metadata.helper_source_sha256.(helper_name) = file_digest(helper_file, 'SHA-256');
end
write_json(fullfile(run_dir, 'status.json'), metadata);
try
    par_base = build_parameters(stack, options);
    save(fullfile(run_dir, 'parameters.mat'), 'par_base', '-v7');
    metadata.parameters = par_base;
    write_json(fullfile(run_dir, 'parameters.json'), metadata);
    cases = {'control', 'hysteretic'};
    for k = 1:numel(cases)
        name = cases{k};
        case_dir = fullfile(run_dir, name);
        mkdir(case_dir);
        par = par_base;
        if strcmp(name, 'control')
            par.taun([1 3]) = 1;
            par.taup([1 3]) = 1;
            par = refresh_reference_device(par, options);
        end
        par.reference_monitor = struct('directory', case_dir, 'case_name', name, ...
            'stage_timeout_s', options.StageTimeout_s, 'run_timeout_s', options.RunTimeout_s, ...
            'run_started', run_started, 'stop_file', fullfile(run_dir, 'STOP'));
        metadata.stage = [name ': initial electrostatics'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        initial = struct('u', 0);
        configured_ionic_species = par.N_ionic_species;
        par.N_ionic_species = 0;
        par.mobset = 0; par.mobseti = 0; par.SRHset = 0; par.radset = 0;
        save(fullfile(case_dir, 'initial-input.mat'), 'par', '-v7');
        sol = stage(initial, par, 1e-12, 0, 0, 0, 20);
        metadata.stage = [name ': electronic relaxation'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        par.mobset = 1; par.SRHset = 1; par.radset = 1;
        sol = stage(sol, par, 1e-3, 0, 0, 0, 40);
        save(fullfile(case_dir, 'electronic.mat'), 'sol', '-v7');
        par.N_ionic_species = configured_ionic_species;
        par.mobseti = 1;
        metadata.stage = [name ': dark zero-bias history'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        if source.settings.seed_dark0_seconds > 0
            sol = stage(sol, par, source.settings.seed_dark0_seconds, 0, 0, 0, 100);
        end
        bias = source.settings.prepare_voltage;
        metadata.stage = [name ': dark prebias history'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        sol = stage(sol, par, source.settings.prepare_seconds, bias, bias, 0, 100);
        prepared = sol;
        save(fullfile(case_dir, 'prepared.mat'), 'prepared', '-v7');
        dwell = source.settings.dv / source.settings.scan_rate;
        if isfield(source.settings, 'light_on_dwell_seconds') && ~isempty(source.settings.light_on_dwell_seconds)
            dwell = source.settings.light_on_dwell_seconds;
        end
        metadata.stage = [name ': illuminated forward scan'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        sol = stage(sol, par, dwell, scan_start, scan_start, 1, 80);
        forward = stage(sol, par, scan_time, scan_start, scan_end, 1, options.ScanPoints);
        save(fullfile(case_dir, 'forward.mat'), 'forward', '-v7');
        sol = forward;
        if source.settings.hold_seconds > 0
            sol = stage(sol, par, source.settings.hold_seconds, scan_end, scan_end, 0, 80);
        end
        metadata.stage = [name ': illuminated reverse scan'];
        write_json(fullfile(run_dir, 'status.json'), metadata);
        fprintf('%s\n', metadata.stage);
        sol = stage(sol, par, dwell, scan_end, scan_end, 1, 80);
        reverse = stage(sol, par, scan_time, scan_end, scan_start, 1, options.ScanPoints);
        save(fullfile(case_dir, 'reverse.mat'), 'reverse', '-v7');
        fwd = export_branch(forward, case_dir, 'fwd');
        rev = export_branch(reverse, case_dir, 'rev');
        result = struct('forward', fwd, 'reverse', rev, ...
            'HI_paper', rev.P_max_W_m2 / fwd.P_max_W_m2 - 1);
        write_json(fullfile(case_dir, 'metrics.json'), result);
        metadata.results.(name) = result;
    end
    metadata.status = 'completed';
    metadata.stage = 'completed';
    write_json(fullfile(run_dir, 'status.json'), metadata);
    fprintf('Completed independent comparison: %s\n', run_dir);
catch failure
    metadata.status = 'failed';
    metadata.error = getReport(failure, 'extended', 'hyperlinks', 'off');
    write_json(fullfile(run_dir, 'status.json'), metadata);
    rethrow(failure);
end
end

function par = build_parameters(stack, options)
% Freeze all dependent pc properties into a plain input structure so manual
% V_bi, ni, trap populations and contact densities can match SolarLab exactly.
obj = pc();
obj.d = [stack.layers.thickness] * 100;
obj.layer_points = options.LayerPoints;
obj.layer_type = {'layer', 'active', 'layer'};
obj.material = {'contact', 'absorber', 'contact'};
obj.xmesh_type = 'linear';
obj.xmesh_coeff = [0.7 0.7 0.7];
obj.T = stack.T;
obj.prob_distro_function = 'Boltz';
obj.Phi_EA = [0 0 0]; obj.Phi_IP = [-1.6 -1.6 -1.6];
obj.Nc = [1e20 1e20 1e20]; obj.Nv = obj.Nc;
obj.Et = [-1.4 -0.8 -0.2];
obj.EF0 = [-1.45 -0.8 -0.15];
obj.Phi_left = -1.45; obj.Phi_right = -0.15;
obj.N_ionic_species = 1;
obj.Nani = [0 0 0]; obj.Ncat = [0 1e19 0];
obj.mu_n = [20 20 20]; obj.mu_p = obj.mu_n;
obj.mu_c = [0 1e-12 0]; obj.mu_a = [0 0 0];
obj.c_max = [1e24 1e24 1e24]; obj.a_max = obj.c_max;
obj.epp = [20 20 20]; obj.B = [1e-10 1e-10 1e-10];
obj.taun = [1 1 1]; obj.taup = obj.taun;
obj.sn = [0 0 0]; obj.sp = [0 0 0];
obj.vsr_zone_loc = {'auto', 'auto', 'auto'};
obj.vsr_mode = 0; obj.vsr_check = 0;
obj.optical_model = 'uniform'; obj.g0 = [0 2.5e21 0];
class_info = metaclass(obj);
par = struct();
for k = 1:numel(class_info.PropertyList)
    property = class_info.PropertyList(k);
    if ~property.Dependent
        par.(property.Name) = obj.(property.Name);
    end
end
dependent = {'active_layer', 'dcell', 'parr', 'd_active', 'dcum', 'dcum0', ...
    'pcum', 'pcum0', 'gamma', 'Eg', 'Efi', 'Vbi'};
for k = 1:numel(dependent), par.(dependent{k}) = obj.(dependent{k}); end
par.e = 1.602176634e-19;
par.kB = 1.380649e-23 / par.e;
par.epp0 = 8.854187817e-12 / par.e / 100;
par.Vbi = stack.V_bi;
par.NA = zeros(1, 3); par.ND = par.NA;
par.ni = zeros(1, 3); par.nt = par.ni; par.pt = par.ni;
par.n0 = zeros(1, 3); par.p0 = par.n0;
for k = 1:3
    material = stack.layers(k).params;
    assert(material.P0_neg == 0 && material.D_ion_neg == 0, 'Only one positive species is supported.');
    assert(material.chi == 0 && material.Eg == 0, 'Only the flat-band Calado parameterization is supported.');
    assert(material.C_n == 0 && material.C_p == 0, 'Auger recombination is not included in this adapter.');
    par.mu_n(k) = material.mu_n * 1e4;
    par.mu_p(k) = material.mu_p * 1e4;
    par.mu_c(k) = material.D_ion * 1e4 / (par.kB * par.T);
    par.Ncat(k) = material.P0 / 1e6;
    par.c_max(k) = material.P_lim / 1e6;
    par.epp(k) = material.eps_r;
    par.B(k) = material.B_rad * 1e6;
    par.taun(k) = material.tau_n; par.taup(k) = material.tau_p;
    par.NA(k) = material.N_A / 1e6; par.ND(k) = material.N_D / 1e6;
    par.ni(k) = material.ni / 1e6;
    par.nt(k) = material.n1 / 1e6; par.pt(k) = material.p1 / 1e6;
    net = par.ND(k) - par.NA(k);
    majority = (abs(net) + hypot(net, 2 * par.ni(k))) / 2;
    minority = par.ni(k)^2 / majority;
    if net >= 0
        par.n0(k) = majority; par.p0(k) = minority;
    else
        par.n0(k) = minority; par.p0(k) = majority;
    end
end
par.n0_l = par.n0(1); par.p0_l = par.p0(1);
par.n0_r = par.n0(end); par.p0_r = par.p0(end);
par.sn_l = 0; par.sp_r = 0;
par.sp_l = options.MajorityVelocity_cm_s;
par.sn_r = options.MajorityVelocity_cm_s;
par.RelTol = options.RelTol; par.AbsTol = options.AbsTol_cm3;
par.int1 = 0; par.int2 = 0;
par.g1_fun_type = 'constant'; par.g1_fun_arg = 0;
par.g2_fun_type = 'constant'; par.g2_fun_arg = 0;
par.mobset = 1; par.mobseti = 1;
par.SRHset = 1; par.radset = 1; par.K_c = 1; par.K_a = 1;
par.Rs = 0;
if options.NeutralContactIonPadding
    assert(all(par.Ncat([1 3]) == 0) && all(par.mu_c([1 3]) == 0), ...
        'Neutral contact padding requires initially ion-free, blocking contacts.');
    par.Ncat([1 3]) = par.Ncat(2);
end
par = refresh_reference_device(par, options);
end

function par = refresh_reference_device(par, options)
par = refresh_device(par);
if options.UniformNodes > 0
    par.xx = linspace(0, par.dcum0(end), options.UniformNodes);
    par.x_sub = getvar_sub(par.xx);
    par.dev = build_device(par, 'whole');
    par.dev_sub = build_device(par, 'sub');
    par.gx1 = generation(par, par.light_source1, par.laser_lambda1);
    par.gx2 = generation(par, par.light_source2, par.laser_lambda2);
end
end

function sol = stage(initial, par, duration, start_bias, end_bias, intensity, samples)
observation = par.reference_monitor;
par = rmfield(par, 'reference_monitor');
remaining_s = observation.run_timeout_s - toc(observation.run_started);
assert(remaining_s > 0, 'SolarLab:ReferenceRunTimeout', 'Reference run time limit exceeded.');
context = struct('case_name', observation.case_name, 'duration_s', duration, ...
    'start_bias_V', start_bias, 'end_bias_V', end_bias, 'intensity', intensity, ...
    'ionic_species', par.N_ionic_species);
monitor = calado_reference_monitor(fullfile(observation.directory, 'latest-stage.json'), ...
    context, min(observation.stage_timeout_s, remaining_s), observation.stop_file, true);
par.tmax = duration; par.tpoints = samples;
% Follow the upstream protocols' six-decade log mesh for long holds while
% retaining the release default 1e-16 s start for the 1e-12 s seed.
par.t0 = min(duration / 100, max(1e-16, duration / 1e6));
par.int1 = intensity; par.g1_fun_arg = intensity;
if start_bias == end_bias
    par.MaxStepFactor = 1;
    par.tmesh_type = 'log10';
    par.V_fun_type = 'constant'; par.V_fun_arg = start_bias;
else
    par.MaxStepFactor = min(1, 0.025 / (0.1 * duration));
    par.tmesh_type = 'linear';
    par.V_fun_type = 'sweep'; par.V_fun_arg = [start_bias end_bias duration];
end
[initial, bias_lift] = calado_reference_bias_lift(initial, par, start_bias);
write_json(fullfile(observation.directory, 'latest-bias-lift.json'), bias_lift);
par.calado_observer = monitor.check;
try
    sol = df_calado_monitored(initial, par);
    sol.par = rmfield(sol.par, 'calado_observer');
    try
        diagnostics = validate_calado_reference_solution(sol);
        write_json(fullfile(observation.directory, 'latest-stage-validation.json'), diagnostics);
    catch invalid_solution
        save(fullfile(observation.directory, 'rejected-stage.mat'), 'sol', '-v7');
        rethrow(invalid_solution);
    end
    monitor.finish('completed');
catch failure
    monitor.finish('failed', failure.identifier);
    rethrow(failure);
end
end

function metrics = export_branch(sol, out_dir, branch)
voltage = dfana.calcVapp(sol);
current = dfana.calcJ(sol, 'sub');
V = voltage(:);
J = -1e4 * current.tot(:, end);
% An independent contact-current read guards against postprocessing errors.
par = sol.par;
left_cond = par.e * par.sp_l * (sol.u(:, 1, 3) - par.p0_l);
field = dfana.calcF(sol, 'sub');
left_disp = -par.e * par.epp0 * par.dev_sub.epp(1) * gradient(field(:, 1), sol.t);
left_total = 1e4 * (left_cond + left_disp);
writetable(table(V, J, left_total, 'VariableNames', ...
    {'V', 'J_active_A_m2', 'J_left_contact_A_m2'}), fullfile(out_dir, [branch '.csv']));
[ordered_v, order] = sort(V);
ordered_j = J(order);
positive = ordered_v >= 0;
power = ordered_v(positive) .* ordered_j(positive);
metrics = struct('P_max_W_m2', max(power), 'J_sc_A_m2', interp1(ordered_v, ordered_j, 0), ...
    'terminal_readout_disagreement_A_m2', max(abs(J - left_total)), ...
    'minimum_n_cm3', min(sol.u(:, :, 2), [], 'all'), ...
    'minimum_p_cm3', min(sol.u(:, :, 3), [], 'all'), ...
    'minimum_c_cm3', min(sol.u(:, :, 4), [], 'all'), ...
    'current_note', 'Main readout: unchanged upstream continuity-based calcJ at right; auxiliary readout: left Robin plus displacement');
inventory = trapz(sol.x, sol.u(:, :, 4), 2);
metrics.ion_inventory_cm2 = inventory;
metrics.ion_inventory_relative_drift = max(abs(inventory - inventory(1))) / max(abs(inventory(1)), realmin);
crossing = find(ordered_j(1:end-1) > 0 & ordered_j(2:end) <= 0, 1);
assert(~isempty(crossing), 'No open-circuit crossing in scan.');
metrics.V_oc_V = interp1(ordered_j(crossing:crossing+1), ordered_v(crossing:crossing+1), 0);
end

function upstream = obtain_release(out_root)
archive = fullfile(out_root, 'Driftfusion-v1.1.1.zip');
if ~isfile(archive)
    fprintf('Downloading official Driftfusion v1.1.1 archive (451.6 kB).\n');
    websave(archive, 'https://zenodo.org/records/6045592/files/barnesgroupICL/Driftfusion-v1.1.1.zip?download=1', weboptions('Timeout', 120));
end
assert(strcmp(file_digest(archive, 'MD5'), 'b8d2fd78d9467967af94c4f617ffea92'), ...
    'Archive checksum mismatch. No downloaded code has been executed.');
% Extract a fresh verified tree so edits in a previous run cannot alter it.
target = tempname(out_root);
upstream = fullfile(target, 'barnesgroupICL-Driftfusion-042bf9e');
unzip(archive, target);
assert(isfile(fullfile(upstream, 'Core', 'df.m')), 'Unexpected release layout.');
end

function digest = file_digest(filename, algorithm)
assert(isfile(filename), 'Checksum input does not exist: %s', filename);
assert(isunix && isfile('/usr/bin/openssl'), ...
    'Java-free file checks require /usr/bin/openssl on macOS or Linux.');
switch algorithm
    case 'MD5'
        flag = '-md5';
        hex_length = 32;
    case 'SHA-256'
        flag = '-sha256';
        hex_length = 64;
    otherwise
        error('Unsupported checksum algorithm: %s', algorithm);
end
% POSIX quoting protects spaces, quotes and shell metacharacters in paths.
% Reading stdin keeps filenames out of the digest output and option parsing.
quote = char(39);
quoted_filename = [quote strrep(char(filename), quote, [quote '"' quote '"' quote]) quote];
[status, output] = system(['/usr/bin/openssl dgst ' flag ' < ' quoted_filename ' 2>&1']);
assert(status == 0, 'Checksum command failed for %s: %s', filename, strtrim(output));
digest = lower(regexp(strtrim(output), '[0-9a-fA-F]+$', 'match', 'once'));
assert(numel(digest) == hex_length, 'Invalid %s checksum output: %s', algorithm, strtrim(output));
end

function write_json(filename, value)
file = fopen(filename, 'w');
assert(file >= 0, 'Cannot open JSON output.');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(value, 'PrettyPrint', true));
end

function restore_environment(old_path, old_dir)
path(old_path);
cd(old_dir);
end
