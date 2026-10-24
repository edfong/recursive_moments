%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run NW model over multiple seeds and time sampling/post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_NW_setting(setting,n_target,seeds,timing_only)

    if nargin < 3 || isempty(seeds)
        seeds = 125:144;
    end
    if nargin < 4
        timing_only = false;
    end

    seeds = seeds(:)';
    script_dir = fileparts(mfilename('fullpath'));
    data_dir = fullfile(script_dir, '..', '..', 'data', 'simulation');
    results_dir = fullfile(script_dir, '..', '..', 'results', 'simulation');
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end

    is_mv = strcmp(setting,"mv");
    
    n_seeds = length(seeds);
    
    posterior_time = zeros(n_seeds,1);
    postproc_time  = zeros(n_seeds,1);
    total_time     = zeros(n_seeds,1);
    
    % Load x_new once if it is common across seeds
    x_new = readmatrix(fullfile(data_dir, ...
        sprintf('sim_%s_x_grid.csv', setting)));
    
    for ss = 1:n_seeds
    
        seed = seeds(ss);
    
        fprintf('\n====================================================\n');
        fprintf('Running seed %d (%d of %d)\n', seed, ss, n_seeds);
        fprintf('====================================================\n');
    
        total_tic = tic;
    
        % Set RNG seed for this run
        rng(seed);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Load Data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        n = n_target;
    
        x = readmatrix(fullfile(data_dir, ...
            sprintf('sim_%s_X_seed%d_n%d.csv', setting, seed, n)));
        Y = readmatrix(fullfile(data_dir, ...
            sprintf('sim_%s_y_seed%d_n%d.csv', setting, seed, n)));
        Y_grid = readmatrix(fullfile(data_dir, ...
            sprintf('sim_%s_y_grid_seed%d_n%d.csv', setting, seed, n)));
    
        % Sample size
        n = size(Y,1);
    
        % Number of discrete covariates
        q = 0;
    
        % Number of continuous covariates
        if is_mv
            p = 5;
        else
            p = 1;
        end
    
        % Pre-compute useful terms
        ny = length(Y_grid);
        dy = Y_grid(2) - Y_grid(1);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% MCMC PARAMETERS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        S = 1000;
        burnin = 3000;
        thin = 2;
    
        mcmc = [burnin,thin,S];
    
        clear S burnin thin
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Base measure hyperparameters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        if is_mv

            % Hyperparameters for the five-covariate multivariate setting
            beta0 = zeros(p + 1, 1);
            alpha1 = 2;
            alpha2 = 0.04054005;
            iC = diag([10, ones(1,p)]) / alpha2 * var(Y);

            mu0 = zeros(p,1);
            c = (1/36) * ones(p,1);
            a1 = 2 * ones(p,1);
            a2 = 0.03747775 * ones(p,1);

        else
    
            % Hyperparameters for the Normal-InverseGamma prior
            beta0 = [5/2,5/8]';
    
            alpha1 = 1;
            alpha2 = 1;
    
            iC = diag([4,1/4]);
    
            % Hyperparameters for the Normal-Gamma prior
            mu0 = [0]';
    
            a1 = [1]';
            a2 = [1]';
    
            c = [1/8]';
    
        end
    
        Hyperparameters = cell(5,1);
        Hyperparameters{1} = beta0;
        Hyperparameters{2} = iC;
        Hyperparameters{3} = [alpha1,alpha2];
        Hyperparameters{4} = [mu0,c,a1,a2];
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Nonparametric prior specification
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        PP = cell(2,1);
        PP{1} = 'StickBreaking';
        PP{2} = [1,1];
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Initial state: randomized alternative
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        cl = 3;
    
        da = discreternd((1:cl)',n)';
    
        ka = 5 * ones(n,1);
    
        Da = cell(n,1);
        for i = 1:n
            Da{i} = discreternd((1:2*cl)',ka(i))';
        end
    
        taua = a1 ./ a2;
        mua = randn(1,2*cl) ./ sqrt(taua .* c) + mu0;
    
        initialState = cell(5,1);
        initialState{1} = da;
        initialState{2} = ka;
        initialState{3} = Da;
        initialState{4} = mua;
        initialState{5} = taua;
    
        clear da ka Da mua taua
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% POSTERIOR SAMPLING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        fprintf('Starting posterior sampling for seed %d...\n', seed);
    
        sampling_tic = tic;
    
        [w,theta,J,d,cputime,lastState] = NPRegNW( ...
            Y,x,q,p,mcmc,Hyperparameters,'Initial',initialState);
    
        posterior_time(ss) = toc(sampling_tic);
    
        fprintf('Posterior sampling time for seed %d: %.2f seconds\n', ...
            seed, posterior_time(ss));
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% POST-PROCESSING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        fprintf('Starting post-processing for seed %d...\n', seed);
    
        postproc_tic = tic;
    
        % Inference for configurations
        [CS,dR,pdR,last_s_dR] = relabel(d);
    
        S = mcmc(3);
    
        % Probability for pointwise credible intervals
        cred_prob = 0.95;
    
        % Predictive mean
        [Y_pred,Y_pred_mat,Y_pred_CI] = NPRegNWprediction( ...
            x_new,q,p,w,theta,J,cred_prob);
    
        % Predictive density estimates
        [Y_fpred,l_fpred,u_fpred] = NPRegNWfprediction( ...
            Y_grid,x_new,q,p,w,theta,J,cred_prob);
    
        % Compute CDF
        CDF = cumsum(Y_fpred, 2) * dy;
    
        % Normalize each row
        CDF = CDF ./ CDF(:, end);
    
        % Compute 95% prediction intervals
        [n_x, n_y] = size(Y_fpred);
    
        idx_25 = zeros(n_x, 1);
        idx_975 = zeros(n_x, 1);
    
        for i = 1:n_x
            [~, idx_25(i)] = min(abs(CDF(i, :) - 0.025));
            [~, idx_975(i)] = min(abs(CDF(i, :) - 0.975));
        end
    
        q_25 = Y_grid(idx_25);
        q_975 = Y_grid(idx_975);
    
        % Prepare prediction dataframe
        pred_NW = table(x_new, Y_pred, q_25, q_975);
        pred_NW.Properties.VariableNames = {'x_plot', 'μ', 'y_l', 'y_h'};
    
        % Evaluate multivariate coverage at the test covariate points.
        x_test = readmatrix(fullfile(data_dir, ...
            sprintf('sim_%s_X_test_seed%d_n%d.csv', setting, seed, n)));
        Y_test = readmatrix(fullfile(data_dir, ...
            sprintf('sim_%s_y_test_seed%d_n%d.csv', setting, seed, n)));
        if is_mv
            x_moments = x_test;
        else
            x_moments = x_new;
        end

        % Moments
        [Y_mean,meanS,Y_mean_CI, ...
        Y_var,varS,Y_var_CI] = ...
        NPRegNWmoments(x_moments,q,p,w,theta,J,cred_prob);
        
        moments_NW = table( ...
        Y_mean, Y_mean_CI(:,3), Y_mean_CI(:,1), Y_mean_CI(:,2), ...
        Y_var, Y_var_CI(:,3), Y_var_CI(:,1), Y_var_CI(:,2));
        
        moments_NW.Properties.VariableNames = { ...
        'μ_mean','μ_med','μ_l','μ_h', ...
        'σ2_mean','σ2_med','σ2_l','σ2_h'};
    
        % Test log PDF
        [pdf_test, logpdf_test, avg_logpdf] = NPRegNWtestpdf(Y_test, x_test, q, p, w, theta, J);

        % Preserve the cluster prediction files during local timing runs.
        if ~timing_only
            writetable(pred_NW, ...
                fullfile(results_dir, ...
                sprintf('sim_%s_seed%d_n%d_pred_NW.csv',setting,seed,n)));
        
            writematrix(Y_fpred, ...
                fullfile(results_dir, ...
                sprintf('sim_%s_seed%d_n%d_pdf_NW.csv',setting,seed,n)));
        
            writetable(moments_NW, ...
                fullfile(results_dir, ...
                sprintf('sim_%s_seed%d_n%d_moments_NW.csv',setting,seed,n)));
        
            writematrix(logpdf_test, ...
                fullfile(results_dir, ...
                sprintf('sim_%s_seed%d_n%d_test_logpdf_NW.csv',setting,seed,n)));
        end

        postproc_time(ss) = toc(postproc_tic);
    
        fprintf('Post-processing time for seed %d: %.2f seconds\n', ...
            seed, postproc_time(ss));
    
        total_time(ss) = toc(total_tic);
    
        fprintf('Total time for seed %d: %.2f seconds\n', ...
            seed, total_time(ss));
    
        % Optional: clear large objects before next seed
        clear w theta J d cputime lastState CS dR pdR last_s_dR
        clear Y_pred Y_pred_mat Y_pred_CI Y_fpred l_fpred u_fpred
        clear CDF q_25 q_975 pred_NW moments_NW
        clear Y_mean meanS Y_mean_CI Y_var varS Y_var_CI
    
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Summarize timing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    runtime_summary = table( ...
        seeds(:), posterior_time, postproc_time, total_time);
    
    runtime_summary.Properties.VariableNames = { ...
        'seed','posterior_sampling_seconds','postprocessing_seconds','total_seconds'};
    
    avg_posterior_time = mean(posterior_time);
    avg_postproc_time = mean(postproc_time);
    avg_total_time = mean(total_time);
    
    fprintf('\n====================================================\n');
    fprintf('Average posterior sampling time: %.2f seconds\n', avg_posterior_time);
    fprintf('Average post-processing time: %.2f seconds\n', avg_postproc_time);
    fprintf('Average total time: %.2f seconds\n', avg_total_time);
    fprintf('====================================================\n');
    
    % Local timing files are separate from the cluster timing files.
    if timing_only
        timing_suffix = '_local';
    else
        timing_suffix = '';
    end

    % Save runtime results
    writetable(runtime_summary, ...
        fullfile(results_dir, ...
        sprintf('sim_%s_n%d_runtime_summary_NW%s.csv', ...
        setting, n_target, timing_suffix)));
    
    avg_runtime_summary = table( ...
        avg_posterior_time, avg_postproc_time, avg_total_time);
    
    avg_runtime_summary.Properties.VariableNames = { ...
        'avg_posterior_sampling_seconds', ...
        'avg_postprocessing_seconds', ...
        'avg_total_seconds'};
    
    writetable(avg_runtime_summary, ...
        fullfile(results_dir, ...
        sprintf('sim_%s_n%d_average_runtime_NW%s.csv', ...
        setting, n_target, timing_suffix)));
end
