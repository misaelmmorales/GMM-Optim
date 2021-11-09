% MISAEL MORALES  -  MATH 7993 Numerical Optimization  -  PROJECT
% ------------------------------------------------------------------------
% OPTIMAL PREDICTION AND CLUSTERING OF GAUSSIAN MIXTURE MODELS
% ------------------------------------------------------------------------
% This is a MTALAB script subroutine to run the Optimization Toolbox
% function to optimize the negative log-likelihood function for the
% Gaussian mixture model.
% 
% Misael Morales - MATH 7993 - July 2020
% ------------------------------------------------------------------------
fprintf('-----------------------------------------------------------\n');
fprintf('Optimization Toolbox subroutine: \n');

% Store as a single output for Optimization Toolbox
x0_ = [mu1_(1),mu1_(2);
      mu2_(1),mu2_(2);
      sigma1_(1,1),sigma1_(1,2);sigma1_(2,1),sigma1_(2,2);
      sigma2_(1,1),sigma2_(1,2);sigma2_(2,1),sigma2_(2,2);
      prop_(1),prop_(2);
      X];

% Optimization Toolbox Options & Settings
options = optimset(...
'Display',     'iter',...
'PlotFcns',    @optimplotfval,...
'MaxIter',     5,...
'MaxFunEvals', 5000,...             % set to a small number
'TolFun',      1e-6,...
'TolX',        1e-10);

% Use the functions in the Optimization Toolbox to optimize the Negative
% Log-Likelihood function for the Gaussian Mixture Model.
[~,optFval,optFlag,optOut]  = fminsearch(@GMM_negloglik,x0_,options);

clear options
