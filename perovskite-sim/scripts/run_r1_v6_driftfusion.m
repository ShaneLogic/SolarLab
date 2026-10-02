function run_r1_v6_driftfusion(requestPath)
% Run bounded V6 independent-reference requests without changing Driftfusion.
% Each request supplies an original input CSV, tolerances, MaxStepFactor,
% ramp, and optional transport coefficient fault. Both controls use the same
% newly prepared soleq.ion. No calcJ/calcJdd current enters the comparison.
req = jsondecode(fileread(requestPath));
maxNumCompThreads(1);
root = req.driftfusionRoot;
addpath(root); addpath(genpath(fullfile(root,'Core')));
addpath(genpath(fullfile(root,'Protocols')));
addpath(genpath(fullfile(root,'Libraries')));
addpath(genpath(fullfile(root,'Scripts')));
addpath(genpath(fullfile(root,'Helper')));
addpath(genpath(fullfile(root,'Optical')));
addpath(genpath(fullfile(root,'Analysis')));
set(0,'DefaultFigureVisible','off');
if ~exist(req.output,'dir'); mkdir(req.output); end
for ii = 1:numel(req.cases)
    if iscell(req.cases); item=req.cases{ii}; else; item=req.cases(ii); end
    destination = fullfile(req.output,[item.tag '.json']);
    assert(~exist(destination,'file'), 'Refuse to overwrite existing case');
    started = tic;
    rep = struct('schema','R1V6DriftfusionRunV1','request',item, ...
        'scope','development_cross_code_reference','status','running');
    rep.matlab_version=version;
    rep.matlab_max_computational_threads=maxNumCompThreads;
    rep.runner_sha256=req.runner_sha256;
    rep.contract_sha256=req.contract_sha256;
    try
        par=pc(item.csv); par.prob_distro_function='Boltz';
        par.RelTol=item.relTol; par.AbsTol=item.absTol;
        par.MaxStepFactor=item.maxStepFactor;
        par.Rs=0; par.Rs_initial=0;
        if isfield(item,'kbtCorrection') && item.kbtCorrection
            % Numerical compensation for DF's immutable rounded kB. The
            % physical temperature stays 300 K; preserve the ionic D.
            ktOriginal=par.kB*par.T;
            par.T=300*8.61733326215e-5/par.kB;
            par.mu_c=par.mu_c*ktOriginal/(par.kB*par.T);
        end
        if strcmp(item.fault,'Dn_p1'); par.mu_n=par.mu_n*1.01; end
        if strcmp(item.fault,'Dion_p1'); par.mu_c=par.mu_c*1.01; end
        par=refresh_device(par);
        assert(par.N_ionic_species==1 && par.z_c==1, 'One positive species required');
        soleq=equilibrate(par);
        rep.constants = struct('e_C',par.e,'kB_eV_K',par.kB,'T_K',par.T, ...
            'epp0',par.epp0,'epp',par.epp,'Vbi',par.Vbi);
        rep.preparation='soleq.ion';
        rep.equilibrium=exportProfile(soleq.ion, fullfile(req.output,[item.tag '_eq.csv']));
        for mi=[0 1]
            ctl='B'; if mi==0; ctl='A'; end
            p1=soleq.ion.par;
            p1.tmax=item.ramp_s; p1.t0=0; p1.tmesh_type=1; p1.tpoints=40;
            p1.V_fun_type='sweepAndStill'; p1.V_fun_arg=[0 -0.005 item.ramp_s];
            p1.mobseti=0; p1.MaxStepFactor=item.maxStepFactor;
            s1=df(soleq.ion,p1);
            assert(s1.t(end)==item.ramp_s,'Ramp did not complete');
            rep.(['ramp_' ctl])=exportProfile(s1, fullfile(req.output,[item.tag '_ramp_' ctl '.csv']));
            for tob = item.times_s(:)'
                assert(tob>0 && tob<=1e-4,'Only bounded approved short windows');
                p2=s1.par; p2.V_fun_type='constant'; p2.V_fun_arg(1)=-0.005;
                p2.g1_fun_type='constant'; p2.g1_fun_arg(1)=0; p2.int1=0;
                p2.mobseti=mi; p2.K_c=1; p2.K_a=1;
                p2.tmax=tob; p2.t0=0; p2.tmesh_type=1; p2.tpoints=200;
                s2=df(s1,p2);
                assert(abs(s2.t(end)-tob)<1e-18,'Dwell did not complete');
                key=sprintf('ctl%s_%s',ctl,strrep(sprintf('%.0e',tob),'-','m'));
                rep.(key)=exportProfile(s2, fullfile(req.output,[item.tag '_' key '.csv']));
            end
        end
        rep.status='completed';
    catch ME
        rep.status='failed'; rep.error=getReport(ME,'extended','hyperlinks','off');
    end
    rep.elapsed_s=toc(started);
    fid=fopen(destination,'w'); fprintf(fid,'%s\n',jsonencode(rep,'PrettyPrint',true)); fclose(fid);
    fprintf('V6 DF %s %s %.3fs\n',item.tag,rep.status,rep.elapsed_s);
end
end

function sc = exportProfile(sol,path)
% Exact flux expression in original Core/df.m (PDEPE subinterval midpoint).
assert(size(sol.u,1)==numel(sol.t),'Truncated PDEPE state/time history');
assert(size(sol.u,2)==numel(sol.x),'State/mesh shape mismatch');
assert(all(isfinite(sol.u(:))) && all(isfinite(sol.t(:))) && ...
    all(isfinite(sol.x(:))), 'Nonfinite state/time/mesh history');
par=sol.par; dev=par.dev_sub; u=squeeze(sol.u(end,:,:));
x=sol.x(:); xs=(x(1:end-1)+x(2:end))/2; dx=diff(x);
V=u(:,1); n=u(:,2); p=u(:,3); c=u(:,4);
ns=(n(1:end-1)+n(2:end))/2; ps=(p(1:end-1)+p(2:end))/2;
cs=(c(1:end-1)+c(2:end))/2;
dV=diff(V)./dx; dn=diff(n)./dx; dp=diff(p)./dx; dc=diff(c)./dx;
kt=par.kB*par.T;
Fn=dev.mu_n(:).*ns.*(-dV+dev.gradEA(:)) + ...
    dev.mu_n(:)*kt.*(dn-ns./dev.Nc(:).*dev.gradNc(:));
Fp=dev.mu_p(:).*ps.*(dV-dev.gradIP(:)) + ...
    dev.mu_p(:)*kt.*(dp-ps./dev.Nv(:).*dev.gradNv(:));
Fc=dev.mu_c(:).*(par.z_c*cs.*dV+kt*(dc+cs.*dc./(dev.c_max(:)-cs)));
Jn=par.e*Fn; Jp=-par.e*Fp;
Jion=-par.z_c*par.e*par.mobseti*par.K_c*Fc;
sc=struct('t_s',sol.t(end),'nx',numel(x),'mobseti',par.mobseti, ...
    'Jn_absmax',max(abs(Jn)),'Jion_absmax',max(abs(Jion)), ...
    'phi_left_V',V(1),'phi_right_V',V(end),'theta_max',max(cs./dev.c_max(:)));
sc.state_time_rows=size(sol.u,1); sc.time_rows=numel(sol.t);
sc.state_nodes=size(sol.u,2); sc.finite_history=true;
for nm=[25 50 75 130 150 175]
    sc.(sprintf('Jn_nm%d',nm))=interp1(xs,Jn,nm*1e-7);
    sc.(sprintf('Jp_nm%d',nm))=interp1(xs,Jp,nm*1e-7);
    sc.(sprintf('Jion_nm%d',nm))=interp1(xs,Jion,nm*1e-7);
end
% Node quadrature includes both ends of the mobile domain; retain raw nodes.
active=x<=1e-5*(1+1e-12);
sc.ion_inventory_cm2=trapz(x(active),c(active));
profile=table(xs, (V(1:end-1)+V(2:end))/2, ns, ps, cs, Jn, Jp, Jion, ...
    'VariableNames',{'x_cm','V_V','n_cm3','p_cm3','c_cm3','Jn_A_cm2','Jp_A_cm2','Jion_A_cm2'});
writetable(profile,path);
writetable(table(x,V,n,p,c,'VariableNames',{'x_cm','V_V','n_cm3','p_cm3','c_cm3'}), ...
    strrep(path,'.csv','_nodes.csv'));
end
