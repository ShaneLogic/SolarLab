function filename = prepare_calado_monitored_driver(source_file, destination)
% Create a reversible observation-only copy of the pinned upstream driver.
arguments
    source_file {mustBeTextScalar}
    destination {mustBeTextScalar}
end
source = fileread(source_file);
header = 'function solstruct = df(varargin)';
renamed = 'function solstruct = df_calado_monitored(varargin)';
anchor = '    function [C,F,S] = dfpde(x,t,u,dudx)';
observed = [anchor newline ...
    '        if x == x_sub(1) && isfield(par, ''calado_observer'')' newline ...
    '            par.calado_observer(t);' newline ...
    '        end'];
assert(count(source, header) == 1 && count(source, anchor) == 1, ...
    'SolarLab:UnexpectedReferenceSource', 'Unexpected upstream driver layout.');
modified = strrep(strrep(source, header, renamed), anchor, observed);
restored = strrep(strrep(modified, observed, anchor), renamed, header);
assert(strcmp(source, restored), 'SolarLab:ReferenceInstrumentationChangedCode', ...
    'Observation-only transformation must be exactly reversible.');
filename = fullfile(destination, 'df_calado_monitored.m');
assert(~isfile(filename), 'SolarLab:ReferenceDriverExists', 'Refusing to overwrite a driver copy.');
% Preserve exact source text, including its existing trailing newline.
file = fopen(filename, 'w', 'n', 'UTF-8');
assert(file >= 0, 'Cannot open monitored driver output.');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s', modified);
end
