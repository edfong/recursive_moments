% Run with the BayesXMix code on the MATLAB path.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

% Local DDP timing runs (only saves run-times).
seed = 125;
timing_only = true;
% For full results, replace the above with:
%   seed = 125:144;       % All 20 seeds, matching the cluster runs.
%   timing_only = false; % Save per-seed moments and test log PDFs as well.

run_NW_setting("kurt", 200, seed, timing_only)
run_NW_setting("kurt", 1000, seed, timing_only)
run_NW_setting("mv", 600, seed, timing_only)
