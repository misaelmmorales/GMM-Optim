% MISAEL MORALES  -  MATH 7993 Numerical Optimization  -  PROJECT
% ------------------------------------------------------------------------
% OPTIMAL PREDICTION AND CLUSTERING OF GAUSSIAN MIXTURE MODELS
% ------------------------------------------------------------------------
% This is a MTALAB script subroutine to define the initialization values
% for the EM-algorithm procedure in GMM.m.
% It is based on the GMM.m MATLAB script.
% 
% Misael Morales - MATH 7993 - July 2020
%
% ------------------------------------------------------------------------
numsamples = 1000;          % Number of randomly generated data points

mu1 = [-2 3];               % Mean vector 1
mu2 = [4 -5];               % Mean vector 2

sigma1 = [2 0 ; 0 2];      % Covariance matrix 1
sigma2 = [8 1 ; 1 8];      % Covariance matrix 2

prop = [7 3];               % Proportion of each distribution
prop = prop/sum(prop);      % Normalize proportions as decimals

x0 = [mu1(1),mu1(2);
      mu2(1),mu2(2);
      sigma1(1,1),sigma1(1,2);sigma1(2,1),sigma1(2,2);
      sigma2(1,1),sigma2(1,2);sigma2(2,1),sigma2(2,2);
      prop(1),prop(2)];

% Define the dimensionality and the number of clusters to be estimated.
n = 2;          % The vector lengths equal to the number of dimensions
k = 2;          % The number of clusters.

% Define stopping criterion. 
tol     = 1e-6;        % Stopping criteria 1: tolerance
maxiter = 50;          % Stopping criteria 2: maximum number of iterations