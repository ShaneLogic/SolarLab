function monitor = calado_reference_monitor(log_file, context, wall_limit_s, stop_file, show_progress)
% Observe PDE callbacks and fail explicitly on a cooperative time/stop limit.
arguments
    log_file {mustBeTextScalar}
    context (1,1) struct
    wall_limit_s (1,1) {mustBeNumeric,mustBeReal,mustBeFinite,mustBePositive}
    stop_file {mustBeTextScalar}
    show_progress (1,1) {mustBeA(show_progress, 'logical')} = false
end
started = tic;
last_write_s = -inf;
cancel_requested = false;
progress_window = [];
if show_progress
    window_name = 'Calado reference diagnostic';
    if isfield(context, 'case_name')
        window_name = ['Calado reference: ' char(context.case_name)];
    end
    progress_window = waitbar(0, 'Starting reference stage', 'Name', window_name, ...
        'CreateCancelBtn', @request_cancel, 'CloseRequestFcn', @request_cancel);
end
record = struct('status', 'running', 'context', context, ...
    'wall_limit_s', wall_limit_s, 'wall_elapsed_s', 0, ...
    'latest_trial_time_s', 0, 'pde_evaluations', 0, ...
    'time_note', 'Trial time is not accepted integration progress');
persist();
monitor = struct('check', @check, 'finish', @finish, 'figure', progress_window);

    function check(trial_time)
        record.pde_evaluations = record.pde_evaluations + 1;
        record.latest_trial_time_s = double(trial_time);
        record.wall_elapsed_s = toc(started);
        if cancel_requested || isfile(stop_file)
            record.status = 'cancelled';
            persist();
            error('SolarLab:ReferenceCancelled', 'Reference stop marker detected: %s', stop_file);
        end
        if record.wall_elapsed_s >= wall_limit_s
            record.status = 'timed_out';
            persist();
            error('SolarLab:ReferenceStageTimeout', ...
                'Reference stage exceeded %.3g wall seconds at trial time %.6g s.', ...
                wall_limit_s, trial_time);
        end
        if record.wall_elapsed_s - last_write_s >= 1
            persist();
            drawnow limitrate
        end
    end

    function finish(status, error_id)
        if nargin < 2, error_id = ''; end
        if strcmp(record.status, 'running')
            record.status = char(status);
        end
        record.wall_elapsed_s = toc(started);
        record.error_identifier = char(error_id);
        persist();
        close_progress();
    end

    function persist()
        writestruct(record, log_file);
        trace_file = [char(log_file) '.log'];
        stream = fopen(trace_file, 'a');
        assert(stream >= 0, 'Cannot append reference progress log.');
        cleanup = onCleanup(@() fclose(stream));
        line = sprintf('%s wall=%.3fs trial=%.6gs PDE=%d\n', ...
            record.status, record.wall_elapsed_s, ...
            record.latest_trial_time_s, record.pde_evaluations);
        fprintf(stream, '%s', line);
        fprintf('%s', line);
        if ~isempty(progress_window) && isgraphics(progress_window)
            waitbar(min(1, record.wall_elapsed_s / wall_limit_s), progress_window, ...
                {sprintf('Wall-time budget: %.1f / %.1f s', record.wall_elapsed_s, wall_limit_s), ...
                sprintf('Trial t: %.6g s | PDE evaluations: %d', ...
                    record.latest_trial_time_s, record.pde_evaluations)});
        end
        last_write_s = record.wall_elapsed_s;
    end

    function request_cancel(~, ~)
        cancel_requested = true;
        close_progress();
    end

    function close_progress()
        if ~isempty(progress_window) && isgraphics(progress_window)
            delete(progress_window);
        end
    end
end
