% MISAEL MORALES  -  MATH 7993 Numerical Optimization  -  PROJECT
% ------------------------------------------------------------------------
% OPTIMAL PREDICTION AND CLUSTERING OF GAUSSIAN MIXTURE MODELS
% ------------------------------------------------------------------------
% This is a MTALAB script subroutine to define an initial guess for the EM
% algorithm routine in GMM.m
% It is based on the GMM.m MATLAB script.
% 
% Misael Morales - MATH 7993 - July 2020
%
% ------------------------------------------------------------------------
% A = Means / B = Covariances / C = Proportions
%% 1: Good initial Guesses
%{
% A1: Mu - Randomly select k data points to serve as the initial means.
indeces = randperm(m);
mu = X(indeces(1:k), :);
mu1g = mu(1,:);
mu2g = mu(2,:);

% B1: Sigma - Use overall covariance of the dataset as the initial variance for each cluster.
sigma = []; 
for j = 1:k
    sigma{j} = cov(X);
end
% C1: Proportions - Assign equal PRIOR probabilities to each cluster.
phi = ones(1, k) * (1 / k);
%}
%% 2: Bad Initial Guess - 2D case
%
mu1g     = [0 , 0];
mu2g     = [50 , -50];
mu = [mu1g;mu2g];
sigma{1} = [0.1, 0; 0, 0.1];
sigma{2} = [3000, 0.1; 0.1, 3000];
phi      = [0.99 0.01];
%
