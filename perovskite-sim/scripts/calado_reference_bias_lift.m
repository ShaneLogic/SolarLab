function [initial, diagnostics] = calado_reference_bias_lift(initial, par, bias)
% Enforce a new electrostatic boundary without changing any density state.
arguments
    initial (1,1) struct
    par (1,1) struct
    bias (1,1) {mustBeNumeric,mustBeReal,mustBeFinite}
end
diagnostics = struct('applied', false, 'left_shift_V', 0, 'right_shift_V', 0, ...
    'method', 'Homogeneous Poisson lift; no density change or physical time increment');
if ~isfield(initial, 'x')
    return
end
x = initial.x(:)';
epsilon = par.dev_sub.epp(:)';
assert(numel(epsilon) == numel(x)-1 && all(diff(x) > 0) ...
    && all(isfinite(epsilon)) && all(epsilon > 0), ...
    'SolarLab:InvalidReferenceMesh', 'Invalid mesh/permittivity for electrostatic lift.');
left = -initial.u(end, 1, 1);
right = par.Vbi - bias - initial.u(end, end, 1);
if left == 0 && right == 0
    return
end
resistance = [0, cumsum(diff(x)./epsilon)];
lift = left + (right-left) * resistance/resistance(end);
initial.u(end, :, 1) = initial.u(end, :, 1) + lift;
diagnostics.applied = true;
diagnostics.left_shift_V = left;
diagnostics.right_shift_V = right;
end
