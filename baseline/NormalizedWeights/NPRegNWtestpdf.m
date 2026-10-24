function [pdf_test, logpdf_test, avg_logpdf] = NPRegNWtestpdf(y_test, x_test, q, p, w, theta, J)
% function [pdf_test, logpdf_test, avg_logpdf] = NPRegNWtestpdf(y_test, x_test, q, p, w, theta, J)
%
% Evaluate the posterior predictive density at test points:
%
%       pdf_test(i) = f_hat(y_test(i) | x_test(i,:))
%
% This avoids constructing a large Y_grid by x_new predictive density matrix.
%
% INPUT:
%   y_test: n_test x 1 vector of test responses
%   x_test: n_test x (q+p) matrix of test covariates
%   q: number of discrete covariates
%   p: number of continuous covariates
%   w: cell array of length S of posterior samples of mixture weights
%   theta: cell array grouping posterior samples by mixture parameter
%   J: S x 1 vector of posterior samples of number of mixture components
%      with positive weights
%
% OUTPUT:
%   pdf_test: n_test x 1 vector of posterior mean predictive densities
%             evaluated at the paired test points
%   logpdf_test: n_test x 1 vector of log predictive densities
%   avg_logpdf: scalar average test log predictive density
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
% Initialize
% -------------------------------------------------------------------------

y_test = y_test(:);
n_test = size(x_test, 1);
S = length(J);

if length(y_test) ~= n_test
    error('y_test must have the same number of rows as x_test.');
end

% Model parameters posterior sample
beta = theta{1};
sigma = theta{2};

if q == 0
    % Only continuous variables
    mu = theta{3};
    tau = theta{4};
else
    % Discrete variables
    rho = theta{3};

    if p > 0
        % Continuous variables
        mu = theta{4};
        tau = theta{5};
    end
end

pdf_test = zeros(n_test, 1);

% -------------------------------------------------------------------------
% Evaluate posterior predictive density at paired test points
% -------------------------------------------------------------------------

 for s = 1:S

    Js = J(s);

    % ---------------------------------------------------------------------
    % Compute covariate-dependent weight matrix
    % wx is n_test x Js
    % ---------------------------------------------------------------------

    lg1 = zeros(n_test, Js);
    lg2 = zeros(n_test, Js);

    % Discrete covariate contribution
    if q > 0
        for h = 1:q
            T = size(rho{s}{h}, 1);

            % indicator matrix n_test x T times log probabilities T x Js
            xh = x_test(:, h);

            ind_h = ...
                kron(ones(1, T), xh) == ...
                kron(ones(n_test, 1), (0:T-1));

            lg1 = lg1 + ind_h * log(rho{s}{h});
        end
    end

    % Continuous covariate contribution
    if p > 0
        for h = 1:p
            xh = x_test(:, q + h);

            lg2 = lg2 ...
                - 0.5 * tau(s, h) * ...
                ( ...
                    kron(ones(1, Js), xh) ...
                    - kron(ones(n_test, 1), mu{s}(h, :)) ...
                ).^2;
        end
    end

    % Unnormalized weights
    wx = kron(ones(n_test, 1), w{s}) .* exp(lg1 + lg2);

    % Normalize row-wise
    wx = wx ./ kron(ones(1, Js), sum(wx, 2));

    % ---------------------------------------------------------------------
    % Component-specific predictive means at x_test
    % betax is n_test x Js
    % ---------------------------------------------------------------------

    betax = [ones(n_test, 1), x_test] * beta{s};

    % ---------------------------------------------------------------------
    % Evaluate normal mixture density at y_test(i), not on a grid
    % ---------------------------------------------------------------------

    y_mat = kron(ones(1, Js), y_test);
    sigma_mat = kron(ones(n_test, 1), sqrt(sigma{s}));

    dens_mat = normpdf(y_mat, betax, sigma_mat);

    % Mixture density for each test point
    pdf_s = sum(wx .* dens_mat, 2);

    % Accumulate posterior average
    pdf_test = pdf_test + pdf_s;

end

% Posterior mean predictive density
pdf_test = pdf_test / S;

% Log predictive density
logpdf_test = log(pdf_test);

% Average test log predictive density
avg_logpdf = mean(logpdf_test);

end
