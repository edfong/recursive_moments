function [Y_mean,meanS,Y_mean_CI,Y_var,varS,Y_var_CI] = ...
    NPRegNWmoments(x_new,q,p,w,theta,J,varargin)
% function [Y_mean,meanS,Y_mean_CI,Y_var,varS,Y_var_CI] =
%     NPRegNWmoments(x_new,q,p,w,theta,J,cred_prob)
%
% Compute posterior mean and variance functions for
% the normalized-weights dependent Dirichlet process regression model.
%
% INPUT:
%   x_new: n_new x q+p matrix of new covariates
%   q: number of discrete covariates
%   p: number of continuous covariates
%   w: cell array of length S of posterior samples of mixture weights
%   theta: cell array of posterior samples of mixture parameters
%   J: S x 1 vector of posterior samples of number of active components
%
% OPTIONAL:
%   cred_prob: pointwise credible interval probability, e.g. 0.95
%
% OUTPUT:
%   Y_mean: n_new x 1 posterior mean of conditional mean function
%   meanS: S x n_new posterior samples of conditional mean function
%   Y_mean_CI: n_new x 3 lower, upper, median credible summaries
%
%   Y_var: n_new x 1 posterior mean of conditional variance function
%   varS: S x n_new posterior samples of conditional variance function
%   Y_var_CI: n_new x 3 lower, upper, median credible summaries

%% Initialize

n_new = size(x_new,1);
S = length(J);

beta = theta{1};
sigma = theta{2};

if q == 0
    mu = theta{3};
    tau = theta{4};
else
    rho = theta{3};

    if p > 0
        mu = theta{4};
        tau = theta{5};
    end
end

meanS = zeros(S,n_new);
varS = zeros(S,n_new);

Y_mean_CI = [];
Y_var_CI = [];

%% Loop over posterior samples

for s = 1:S

    Js = J(s);

    % -------------------------------------------------------------
    % Compute covariate-dependent weights
    % -------------------------------------------------------------

    lg1 = zeros(n_new,Js);
    lg2 = zeros(n_new,Js);

    if q > 0
        for h = 1:q
            T = size(rho{s}{h},1);

            lg1 = lg1 + ...
                (kron(ones(1,T),x_new(:,h)) == ...
                 kron(ones(n_new,1),(0:T-1))) * log(rho{s}{h});
        end
    end

    if p > 0
        for h = 1:p
            lg2 = lg2 - 0.5 * tau(s,h) * ...
                (kron(ones(1,Js),x_new(:,q+h)) - ...
                 kron(ones(n_new,1),mu{s}(h,:))).^2;
        end
    end

    w_s = w{s};
    w_s = w_s(:)';          % force row vector
    w_s = w_s(1:Js);        % keep active components only
    
    wx = kron(ones(n_new,1),w_s) .* exp(lg2 + lg1);
    wx = wx ./ kron(ones(1,Js),sum(wx,2));
    
    beta_s = beta{s};
    beta_s = beta_s(:,1:Js);    % keep active components only
    
    betax = [ones(n_new,1),x_new] * beta_s;
    % betax is n_new x Js

    % -------------------------------------------------------------
    % Component-specific variances
    % -------------------------------------------------------------
    % Sigma{s} is interpreted as the component variance.

    sig2 = sigma{s};
    sig2 = sig2(:)';  % force row vector

    if length(sig2) == 1
        sig2 = repmat(sig2,1,Js);
    else
        sig2 = sig2(1:Js);
    end

    sig2_mat = kron(ones(n_new,1),sig2);

    % -------------------------------------------------------------
    % Conditional mean for this posterior draw
    % -------------------------------------------------------------
    
    mu_y = sum(wx .* betax,2);
    % n_new x 1

    meanS(s,:) = mu_y';

    % -------------------------------------------------------------
    % Conditional raw second moment E[Y^2]
    %
    % For a normal component Y_j ~ N(m_j, sig2_j):
    %
    %   E[Y_j^2] = sig2_j + m_j^2
    % -------------------------------------------------------------

    EY2 = sum(wx .* (sig2_mat + betax.^2),2);

    % -------------------------------------------------------------
    % Conditional variance
    % -------------------------------------------------------------

    var_y = EY2 - mu_y.^2;

    % Numerical guard against tiny negative values due to floating error.
    var_y(var_y < 0 & var_y > -1e-10) = 0;

    varS(s,:) = var_y';

end

%% Posterior means of the functions

Y_mean = mean(meanS,1)';
Y_var = mean(varS,1)';

%% Credible intervals

if ~isempty(varargin)

    cred_prob = varargin{1};

    alpha = (1 - cred_prob) / 2;

    lower_idx = max(1,ceil(S * alpha));
    upper_idx = min(S,ceil(S * (1 - alpha)));
    median_idx = max(1,ceil(S * 0.5));

    % Mean credible intervals
    meanS_sort = sort(meanS,1);
    mean_l = meanS_sort(lower_idx,:);
    mean_u = meanS_sort(upper_idx,:);
    mean_m = meanS_sort(median_idx,:);

    Y_mean_CI = [mean_l',mean_u',mean_m'];

    % Variance credible intervals
    varS_sort = sort(varS,1);
    var_l = varS_sort(lower_idx,:);
    var_u = varS_sort(upper_idx,:);
    var_m = varS_sort(median_idx,:);

    Y_var_CI = [var_l',var_u',var_m'];

end

end
