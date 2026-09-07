classdef CaladoReferenceMonitorTest < matlab.unittest.TestCase
    % Observer contracts only; these tests do not certify the physical solver.
    properties
        WorkFolder
    end
    methods (TestClassSetup)
        function sourcePath(testCase)
            source = fileparts(fileparts(mfilename('fullpath')));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(source));
        end
    end
    methods (TestMethodSetup)
        function temporaryFolder(testCase)
            testCase.applyFixture(matlab.unittest.fixtures.WorkingFolderFixture);
            testCase.WorkFolder = pwd;
        end
    end
    methods (Test)
        function completedStageHasReadableProgress(testCase)
            filename = fullfile(testCase.WorkFolder, 'progress.json');
            monitor = calado_reference_monitor(filename, struct('duration_s', 2), ...
                30, fullfile(testCase.WorkFolder, 'STOP'));
            monitor.check(0.25);
            monitor.finish('completed');
            record = readstruct(filename);
            testCase.verifyEqual(record.status, "completed");
            testCase.verifyEqual(record.latest_trial_time_s, 0.25, AbsTol=1e-15);
            testCase.verifyEqual(record.pde_evaluations, 1, AbsTol=0);
            testCase.verifyTrue(isfile([filename '.log']));
        end

        function timeoutFailsExplicitly(testCase)
            filename = fullfile(testCase.WorkFolder, 'progress.json');
            monitor = calado_reference_monitor(filename, struct(), realmin, ...
                fullfile(testCase.WorkFolder, 'STOP'));
            testCase.verifyError(@() monitor.check(0), 'SolarLab:ReferenceStageTimeout');
            monitor.finish('failed', 'SolarLab:ReferenceStageTimeout');
            record = readstruct(filename);
            testCase.verifyEqual(record.status, "timed_out");
        end

        function stopMarkerCancelsOnlyThisRun(testCase)
            filename = fullfile(testCase.WorkFolder, 'progress.json');
            stopFile = fullfile(testCase.WorkFolder, 'STOP');
            monitor = calado_reference_monitor(filename, struct(), 30, stopFile);
            writelines("stop", stopFile);
            testCase.verifyError(@() monitor.check(0), 'SolarLab:ReferenceCancelled');
            record = readstruct(filename);
            testCase.verifyEqual(record.status, "cancelled");
        end

        function copiedDriverPreservesSourceAndArithmetic(testCase)
            sourceFile = fullfile(testCase.WorkFolder, 'df.m');
            writelines(fixtureSource(), sourceFile);
            original = fileread(sourceFile);
            filename = prepare_calado_monitored_driver(sourceFile, testCase.WorkFolder);
            result = df_calado_monitored(struct('u', 0), struct());
            testCase.verifyEqual(result.value, 7, AbsTol=0);
            testCase.verifyEqual(fileread(sourceFile), original);
            testCase.verifyTrue(isfile(filename));
        end

        function copiedDriverInvokesObserver(testCase)
            sourceFile = fullfile(testCase.WorkFolder, 'df.m');
            writelines(fixtureSource(), sourceFile);
            prepare_calado_monitored_driver(sourceFile, testCase.WorkFolder);
            monitor = calado_reference_monitor(fullfile(testCase.WorkFolder, 'progress.json'), ...
                struct(), realmin, fullfile(testCase.WorkFolder, 'STOP'));
            parameters = struct('calado_observer', monitor.check);
            testCase.verifyError(@() df_calado_monitored(struct('u', 0), parameters), ...
                'SolarLab:ReferenceStageTimeout');
        end

        function unexpectedSourceFailsClosed(testCase)
            sourceFile = fullfile(testCase.WorkFolder, 'unexpected.m');
            writelines("function out = unknown; out = 0; end", sourceFile);
            testCase.verifyError(@() prepare_calado_monitored_driver(sourceFile, testCase.WorkFolder), ...
                'SolarLab:UnexpectedReferenceSource');
        end

        function progressWindowCloseRequestsCancellation(testCase)
            filename = fullfile(testCase.WorkFolder, 'progress.json');
            monitor = calado_reference_monitor(filename, struct(), 30, ...
                fullfile(testCase.WorkFolder, 'STOP'), true);
            testCase.addTeardown(@() monitor.finish('failed', 'TestCleanup'));
            testCase.verifyTrue(isgraphics(monitor.figure));
            close(monitor.figure);
            testCase.verifyError(@() monitor.check(0), 'SolarLab:ReferenceCancelled');
            testCase.verifyFalse(isgraphics(monitor.figure));
        end

        function validReferenceAllowsToleranceScaleNegativity(testCase)
            sol = fixtureSolution();
            sol.u(2, 2, 4) = -1e-7;
            result = validate_calado_reference_solution(sol);
            testCase.verifyEqual(result.density_minima_cm3(3), -1e-7, AbsTol=1e-15);
        end

        function negativeReferenceFailsClosed(testCase)
            sol = fixtureSolution();
            sol.u(2, 2, 4) = -1e18;
            testCase.verifyError(@() validate_calado_reference_solution(sol), ...
                'SolarLab:NegativeReferenceDensity');
        end

        function truncatedReferenceFailsClosed(testCase)
            sol = fixtureSolution();
            sol.u = sol.u(1:2, :, :);
            testCase.verifyError(@() validate_calado_reference_solution(sol), ...
                'SolarLab:InvalidReferenceSolution');
        end

        function nonfiniteReferenceFailsClosed(testCase)
            sol = fixtureSolution();
            sol.u(2, 2, 2) = nan;
            testCase.verifyError(@() validate_calado_reference_solution(sol), ...
                'SolarLab:InvalidReferenceSolution');
        end

        function biasLiftChangesOnlyLastPotential(testCase)
            sol = fixtureSolution();
            sol.u(:,:,2:4) = 3;
            sol.u(end,:,1) = [0, 0.1, 0.9, 1.3];
            par = struct('Vbi', 1.3, 'dev_sub', struct('epp', [10 20 40]));
            [lifted, diagnostics] = calado_reference_bias_lift(sol, par, -1);
            delta = lifted.u(end,:,1)-sol.u(end,:,1);
            flux = par.dev_sub.epp .* diff(delta) ./ diff(sol.x);
            testCase.verifyEqual(lifted.u(:,:,2:4), sol.u(:,:,2:4), AbsTol=0);
            testCase.verifyEqual(lifted.u(1:2,:,1), sol.u(1:2,:,1), AbsTol=0);
            testCase.verifyEqual(lifted.u(end,end,1), 2.3, AbsTol=1e-14);
            testCase.verifyEqual(lifted.u(end,1,1), 0, AbsTol=1e-14);
            testCase.verifyEqual(flux, repmat(flux(1), size(flux)), AbsTol=1e-13);
            testCase.verifyTrue(diagnostics.applied);
        end

        function unchangedBiasPreservesInitialState(testCase)
            sol = fixtureSolution();
            sol.u(end,:,1) = [0, 0.1, 0.9, 1.3];
            par = struct('Vbi', 1.3, 'dev_sub', struct('epp', [20 20 20]));
            [lifted, diagnostics] = calado_reference_bias_lift(sol, par, 0);
            testCase.verifyEqual(lifted, sol);
            testCase.verifyFalse(diagnostics.applied);
        end
    end
end

function sol = fixtureSolution()
sol = struct('u', zeros(3, 4, 4), 't', [0 0.1 0.2], 'x', linspace(0, 1, 4), ...
    'par', struct('N_ionic_species', 1, 'AbsTol', 1e-6));
end

function text = fixtureSource()
text = ["function solstruct = df(varargin)"; ...
    "par = varargin{2};"; "x_sub = 0;"; ...
    "[C,F,S] = dfpde(0,0,3,4);"; "solstruct = struct('value', C+F+S);"; ...
    "    function [C,F,S] = dfpde(x,t,u,dudx)"; ...
    "        C=u; F=dudx; S=x+t;"; "    end"; "end"];
end
