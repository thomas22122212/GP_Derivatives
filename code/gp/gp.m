% Interface for training a GP using the SE Kernel
% 
% [mu, K, dmu] = gp_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix
% dmu: dmean function handle such that calling dmu(XX,co) for some predictive points XX calculates the derivation of mean of the GP at XX wrt the coordinates co. e.g. dmu(XX,[2;3]) derives a three dimensional GP after the secound and the third dimension/coordinate of the GP

function [mu, K, dmu] = gp(X, Y)
[ntrain, d] = size(X);

% Initial hyperparameters
ell0 = 0.5*sqrt(d);
s0 = std(Y);
sig0 = 5e-2*s0; 
beta = 1e-6;

% Train GP 
cov = @(hyp) se_kernel(X, hyp);
lmlfun = @(x) lml_exact(cov,Y, x, beta);
hyp = struct('cov', log([ell0, s0]), 'lik', log([sig0]));
params = minimize_quiet(hyp, lmlfun, -50);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE with gradients: (ell, s, sigma1) = (%.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Calculate interpolation coefficients
sigma2 = sigma^2*ones(1, ntrain);
K = se_kernel(X, params) + diag(sigma2);
lambda = K\Y;

% Function handle returning GP mean to be output
mu = @(XX) mean(XX, X, lambda, params);
dmu = @(XX, co) dmean(XX, X, lambda, params, co);
end

function ypred = mean(XX, X, lambda, params)
KK = se_kernel(X, params, XX);
ypred = KK*lambda;
end

function dpred = dmean(XX, X, lambda, params, coordinate)
arguments
    XX
    X
    lambda
    params
    coordinate (:,1) double = 0;
end

if coordinate == 0
    dpred = mean(XX, X, lambda, params);
else
    counter = zeros(size(XX,2),1);
    for i = 1:size(coordinate,1)
        counter(coordinate(i,1),1) = counter(coordinate(i,1),1) + 1;
    end
    
    f = ones(size(XX,1),size(X,1));
    for i = 1:size(XX,2)
        switch counter(i,1)
            case 1
                f = f.* factor1derivation(XX, X, params, i);
            case 2
                f = f.* factor2derivations(XX, X, params, i);
            case 3
                f = f.* factor3derivations(XX, X, params, i);
        end
    end
    KK = se_kernel(X, params, XX);
    KK = KK.*f;
    dpred = KK*lambda;
end
end

function f = factor1derivation(XX, X, params, coordinate)
f = (transpose(X(:,coordinate))-XX(:,coordinate))/(exp(2*params.cov(1)));
end

function f = factor2derivations(XX, X, params, coordinate)
f = (transpose(X(:,coordinate))-XX(:,coordinate)).^2/(exp(2*params.cov(1)))^2-1/exp(2*params.cov(1));
end

function f = factor3derivations(XX, X, params, coordinate)
f = (transpose(X(:,coordinate))-XX(:,coordinate)).^3/(exp(2*params.cov(1)))^3-3*(transpose(X(:,coordinate))-XX(:,coordinate))/(exp(2*params.cov(1)))^2;
end
