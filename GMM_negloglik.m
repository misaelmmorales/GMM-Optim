function nLL = GMM_negloglik(xstart) 
%      X - the random multivariate Gaussian mixture data
% xstart - the starting parameters: means, covariances, proportions (from EM)
%
% Define parameters for the optimization problem
Mu{1}    = xstart(1,:);
Mu{2}    = xstart(2,:);
Sigma{1} = xstart(3:4,:);
Sigma{2} = xstart(5:6,:);
Alpha    = xstart(7,:);

% Define constant parameters
%data = xstart(8:length(xstart),:);
global X

% Compute Gaussian Mixture & negative Log-Likelihood
nLL = -sum(log(...
    Alpha(1)/sum(Alpha)*GaussianNormalDist(X,Mu{1}, chol(Sigma{1})*chol(Sigma{1})') + ...
    Alpha(2)/sum(Alpha)*GaussianNormalDist(X,Mu{2}, chol(Sigma{2})*chol(Sigma{2})') ));

end
