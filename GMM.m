% MISAEL MORALES  -  MATH 7993 Numerical Optimization  -  PROJECT
% ------------------------------------------------------------------------
% OPTIMAL PREDICTION AND CLUSTERING OF GAUSSIAN MIXTURE MODELS
% ------------------------------------------------------------------------
% This is a MTALAB script routine to perform the Expectation-Maximization
% algorithm on a multivariate gaussian mixture model.
% The goal is to obtain estimates for the means, covariances, and
% proportions and to further improve estimates using the Optimization
% Toolbox.
% Misael Morales - MATH 7993 - July 2020
%
% ------------------------------------------------------------------------
clear; clc; close all
tic % Start timer for script runtime
fprintf('MISAEL MORALES - MATH 7993 NUMERICAL OPTIMIZATION - PROJECT\n');
fprintf('OPTIMIZATION OF MULTIVARIATE GAUSSIAN MIXTURE MODEL\n');
% ------------------------------------------------------------------------
%% Step 0: Define problem variables
numsamples = 1000;          % Number of randomly generated data points

mu1 = [-2 3];               % Mean vector 1
mu2 = [4 -5];               % Mean vector 2
sigma1 = [2 1 ; 0 5];       % Covariance matrix 1
sigma2 = [8 1 ; 1 8];       % Covariance matrix 2
prop = [7 3];               % Proportion of each distribution
prop = prop/sum(prop);      % Normalize proportions as decimals constraint

x0 = [mu1(1),mu1(2);mu2(1),mu2(2);
      sigma1(1,1),sigma1(1,2);sigma1(2,1),sigma1(2,2);
      sigma2(1,1),sigma2(1,2);sigma2(2,1),sigma2(2,2);
      prop(1),prop(2)];

% Define the dimensionality and the number of clusters to be estimated.
n = 2;          % The vector lengths equal to the number of dimensions
k = 2;          % The number of clusters.

% Define stopping criterion. 
tol     = 1e-6;        % Stopping criteria 1: tolerance
maxiter = 5;           % Stopping criteria 2: maximum number of iterations
%
%% Step 1: Print data & Generate data from TWO 2D distributions.

% Print initial variables
fprintf('Real Mu1: [%.2f, %.2f]\n', mu1);
fprintf('Real Mu2: [%.2f, %.2f]\n', mu2);
fprintf('Real Sigma1: [%.2f %.2f ; %.2f %.2f]\n', sigma1);
fprintf('Real Sigma2: [%.2f %.2f ; %.2f %.2f]\n', sigma2);
fprintf('Real Proportions: [%.3f, %.3f]\n', prop);

dim1 = numsamples*prop(1);    % Number of data points from each distribution
dim2 = numsamples*prop(2);

% Generate sample points with the specified means and covariance matrices.
R1 = chol(sigma1);
X1 = randn(dim1, 2) * R1;     % Random normal (randn) number generator
X1 = X1 + repmat(mu1, size(X1, 1), 1);  % n*1 repmat for same dims as X1

R2 = chol(sigma2);
X2 = randn(dim2, 2) * R2;     % Random normal (randn) number generator
X2 = X2 + repmat(mu2, size(X2, 1), 1);  % n*1 repmat for same dims as X2

global X
X = [X1; X2]; % Combine the two sets of random normal elements into one


m = size(X, 1); % The number of data points; rows of X

% Ensure positive-semidefinite constraint on covariance matrices
sigma1 = chol(sigma1)*chol(sigma1)';
sigma2 = chol(sigma2)*chol(sigma2)';

% Generate a 1,2 classification to use Classification Learner Toolbox
class1 = cat(2, X1,   ones(size(X1,1), 1));
class2 = cat(2, X2, 2*ones(size(X2,1), 1));
Xclass = [class1;class2];

%%====================================================
%% STEP 2: Define initial values for the parameters.
% Use the GMM_InitialGuess MATLAB subroutine to define the starting value
% for the EM-algorithm.
%
% A = Means / B = Covariances / C = Proportions
%% 1a: Good initial Guesses
%{
% A1: Mu - Randomly modify by Normal Distribution the means to be initial guess.
mu = [mu1+rand(1);mu2+rand(1)];
mu1g = mu(1,:); %for printing initial guesses
mu2g = mu(2,:); %for printing initial guesses

% B1: Sigma - Use overall covariance of the dataset as the initial variance for each cluster.
sigma = []; 
for j = 1:k
    sigma{j} = cov(X);
end
% C1: Proportions - Assign equal PRIOR probabilities to each cluster.
phi = ones(1, k) * (1 / k);
%}
%% 1b: Bad Initial Guess - 2D case
%
mu1g     = [-2 , 2];
mu2g     = [5 , -5];
mu = [mu1g;mu2g];
sigma{1} = [50, 5; 5, 50];
sigma{2} = [30, 0.1; 1, 3];
phi      = [99 11];
phi      = phi/sum(phi);

x0g = [mu1g; mu2g;
       sigma{1}(1,:);sigma{1}(2,:);
       sigma{2}(1,:);sigma{2}(2,:);
       phi(1),phi(2)];
%}
%%
fprintf('-----------------------------------------------------------\n');
fprintf('Initial Guess Mu1: [%.2f, %.2f]\n', mu1g);
fprintf('Initial Guess Mu2: [%.2f, %.2f]\n', mu2g);
fprintf('Initial Guess Sigma1: [%.2f %.2f ; %.2f %.2f]\n', sigma{1});
fprintf('Initial Guess Sigma2: [%.2f %.2f ; %.2f %.2f]\n', sigma{2});
fprintf('Initial Guess Proportions: [%.3f, %.3f]\n', phi);
%% Scatterplot and Contours for Original Data
% Define parameters for grid density and dimensions.
gridSize = 50;
 maxgrid = max(max(mu1,mu2)) + max(max(max(sigma1,sigma2)));
       u = linspace(-maxgrid, maxgrid, gridSize);
% Display a scatter plot of the data from the two distributions.
% Save space as a subplot to plot side-by-side with EM-generated
% distribution estimates.
figure(1)
subplot(2,3,1);
hold off;
plot(X1(:, 1), X1(:, 2), 'g.');
hold on;
plot(X2(:, 1), X2(:, 2), 'm.');

% Define a [gridSize^2 x 2] matrix 'gridX' of coordinates representing
% the input values over the grids.
[A , B] = meshgrid(u, u);
gridX   = [A(:), B(:)];

% Calculate the Gaussian distribution for every value in the grid.
z1 = GaussianNormalDist(gridX, mu1, sigma1);
z2 = GaussianNormalDist(gridX, mu2, sigma2);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines (5) to show the pdf over the data.
contour(u, u, Z1, 5);
contour(u, u, Z2, 5);
daspect([1 1 1])

plot(mu1(1),mu1(2),'k.','MarkerSize',20)    %plot original center X1
plot(mu2(1),mu2(2),'k.','MarkerSize',20)    %plot original center X2

title('True PDFs');
%% Histogram for Original Data
figure(2)
subplot(2,1,1);
hold off
histogram(X);
hold on
title('Histogram of MV Normal Data')
xlabel('X'); ylabel('Frequency');
%savefig('GMM_histogram')
% Create a 3D Surface Plot for the pdf response function
subplot(2,1,2);
hold off
surf([Z1;Z2])
hold on
title('MV PDF for Generated Data')
%% Scatterplot and Contours of Initial Estimates
figure(1)
subplot(2,3,2);
hold off;
plot(X1(:, 1), X1(:, 2), 'g.');
hold on;
plot(X2(:, 1), X2(:, 2), 'm.');
% Calculate the Gaussian distribution for every value in the grid.
z1 = GaussianNormalDist(gridX, mu1g, sigma{1});
z2 = GaussianNormalDist(gridX, mu2g, sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines (5) to show the pdf over the data.
contour(u, u, Z1, 5);
contour(u, u, Z2, 5);
daspect([1 1 1])

plot(mu1g(1),mu1g(2),'k.','MarkerSize',20)    %plot original center X1
plot(mu2g(1),mu2g(2),'k.','MarkerSize',20)    %plot original center X2

title('Initial Guess');
%%===================================================
%% STEP 3: Run Expectation Maximization (EM) Algorithm
for iter = 1:maxiter    
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k);
    
    % For each Gaussian...
    for j = 1:k
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = GaussianNormalDist(X, mu(j, :), sigma{j});
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    %%===============================================
    %% STEP 3b: Maximization
    %%% Calculate the probability for each data point for each distribution.

    % Store the previous means, to be optimized iteratively.
    prevMu = mu;    
    
    % For each of the Gaussian...
    for j = 1:k
    
        % Calculate the prior probability for cluster 'j'.
        prop_(j) = mean(W(:, j), 1);
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        weights  = W(:,j);
        val      = weights' * X;
        mu(j, :) = val ./ sum(weights, 1);

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        sigma_k = zeros(n, n);
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        for i = 1:m
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
   
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
    % Check for convergence.
    if abs(mu - prevMu) < tol
        break
    end
            
% End of Expectation Maximization (EM)    
end

%% Step 4: Print final values
fprintf('-----------------------------------------------------------\n');
% Rename final variables
mu1_    = mu(1,:);
mu2_    = mu(2,:);
sigma1_ = sigma{1};
sigma2_ = sigma{2};

% print final variables
fprintf('EM-estimated Mu1: [%.4f, %.4f]\n', mu1_);
fprintf('EM-estimated Mu2: [%.4f, %.4f]\n', mu2_);
fprintf('EM-estimated Sigma1: [%.3f %.3f ; %.3f %.3f]\n', sigma1_);
fprintf('EM-estimated Sigma2: [%.3f %.3f ; %.3f %.3f]\n', sigma2_);
fprintf('EM-estimated Proportions: [%.4f, %.4f]. Sum: %.4f\n', prop_, sum(prop_));

% Print the stopping criteria and the total number of EM iterations
fprintf('Stopping Criteria: max iterations (%d) | tolerance (%.1d)\n', maxiter, tol)

if iter == maxiter
    fprintf('Maximum number of iterations (%d) was reached before convergence!\n', iter)
else 
    fprintf('Total EM Iterations: %d\n', iter);
end

% END of main script
%% Scatterplot and Contours for EM-estimated Data
%
% Display a scatter plot of the data from the two distributions.
figure(1)
subplot(2,3,4);
hold off;
plot(X1(:, 1), X1(:, 2), 'g.');
hold on;
plot(X2(:, 1), X2(:, 2), 'm.');

% Again, create a [gridSize^2 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
[A , B] = meshgrid(u, u);
gridX   = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1_ = GaussianNormalDist(gridX, mu1_, sigma1_);
z2_ = GaussianNormalDist(gridX, mu2_, sigma2_);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1_ = reshape(z1_, gridSize, gridSize);
Z2_ = reshape(z2_, gridSize, gridSize);

% Plot the contour lines (5) to show the pdf over the data.
contour(u, u, Z1_, 5);
contour(u, u, Z2_, 5);
daspect([1 1 1])

plot(mu1(1),mu1(2),'k.','MarkerSize',20)    %plot original center X1
plot(mu2(1),mu2(2),'k.','MarkerSize',20)    %plot original center X2
plot(mu1_(1),mu1_(2),'k*')  %plot new center X1
plot(mu2_(1),mu2_(2),'k*')  %plot new center X2
title('EM Contours');
%savefig('GMM_orig_EM_data_contour')
%
% ------------------------------------------------------------------------
%% Optimization Toolbox - fminsearch
fprintf('-----------------------------------------------------------\n');
fprintf('Optimization Toolbox subroutine: \n');

% Store as a single output for Optimization Toolbox
x0_ = [mu1_(1),mu1_(2); mu2_(1),mu2_(2);
      sigma1_(1,1),sigma1_(1,2); sigma1_(2,1),sigma1_(2,2);
      sigma2_(1,1),sigma2_(1,2); sigma2_(2,1),sigma2_(2,2);
      prop_(1),prop_(2)];
% Optimization Toolbox Options & Settings
%'Display',     'iter',...
options = optimset(...
'PlotFcns',    {@optimplotfval,@optimplotfunccount},...
'MaxIter',     500,...
'MaxFunEvals', 5000,...             
'TolFun',      1e-6,...
'TolX',        1e-6);
% Use the functions in the Optimization Toolbox to optimize the Negative
% Log-Likelihood function for the Gaussian Mixture Model.
figure(3)
[optXval,optFval,optFlag,optOut]  = fminsearch(@GMM_negloglik, x0_, options);
copyobj(get(gcf,'children'),figure(3));
hold on
xlabel('fminsearch iteration')
hold off
optXval(7,:) = optXval(7,:)/sum(optXval(7,:));
mu1_opt1 = optXval(1,:);
mu2_opt1 = optXval(2,:);
sigma1_opt1 = optXval(3:4,:);
sigma2_opt1 = optXval(5:6,:);
prop_opt1 = optXval(7,:);
% Printing optimization 1 outputs      
fprintf('-----------------------------------------------------------\n');
fprintf('NMS Optimization-estimated Mu1: [%.4f, %.4f]\n', mu1_opt1);
fprintf('NMS Optimization-estimated Mu2: [%.4f, %.4f]\n', mu1_opt1);
fprintf('NMS Optimization-estimated Sigma1: [%.3f %.3f ; %.3f %.3f]\n', sigma1_opt1);
fprintf('NMS Optimization-estimated Sigma2: [%.3f %.3f ; %.3f %.3f]\n', sigma2_opt1);
fprintf('NMS Optimization-estimated Proportions: [%.4f, %.4f]. Sum: %.3f \n', prop_opt1, sum(prop_opt1));   
fprintf('-----------------------------------------------------------\n');
%
%% Printing optimization 2 outputs - fminunc
% Optimization Toolbox Options & Settings
%'Display',     'iter',...
options2 = optimset(...
'PlotFcns',    {@optimplotfval,@optimplotfirstorderopt}, ...
'MaxIter',     5000,...       
'MaxFunEvals', 1E6,...                 
'TolFun',      1e-15,...
'TolX',        1e-15);
% Use the functions in the Optimization Toolbox to optimize the Negative
% Log-Likelihood function for the Gaussian Mixture Model.
figure(4)
[optXval2,optFval2,optFlag2,optOut2]  = fminunc(@GMM_negloglik, x0_, options2);
copyobj(get(gcf,'children'),figure(4));
hold on
xlabel('fminunc iteration')
hold off
optXval2(7,:) = optXval2(7,:)/sum(optXval2(7,:));
mu1_opt2 = optXval2(1,:);
mu2_opt2 = optXval2(2,:);
sigma1_opt2 = optXval2(3:4,:);
sigma2_opt2 = optXval2(5:6,:);
prop_opt2 = optXval2(7,:);
% Printing optimization 2 outputs      
fprintf('-----------------------------------------------------------\n');
fprintf('QN Optimization-estimated Mu1: [%.4f, %.4f]\n', mu1_opt2);
fprintf('QN Optimization-estimated Mu2: [%.4f, %.4f]\n', mu2_opt2);
fprintf('QN Optimization-estimated Sigma1: [%.3f %.3f ; %.3f %.3f]\n', sigma1_opt2);
fprintf('QN Optimization-estimated Sigma2: [%.3f %.3f ; %.3f %.3f]\n', sigma2_opt2);
fprintf('QN Optimization-estimated Proportions: [%.4f, %.4f]. Sum: %.3f\n', prop_opt2,sum(prop_opt2));  
%
fprintf('-----------------------------------------------------------\n');
fprintf('EM-estimated Negative Log-Likelihood:        %.4f \n', GMM_negloglik(x0_));
fprintf('fminsearch Negative Log-Likelihood:          %.4f \n', optFval);
fprintf('fminunc Negative Log-Likelihood:             %.4f \n', optFval2);
fprintf('Original Parameters Negative Log-Likelihood: %.4f \n', GMM_negloglik(x0));
%% Plot the new refined estimates
figure(1)
% NMS Optimtool Estimates
subplot(2,3,5);
hold off;
plot(X1(:, 1), X1(:, 2), 'g.');
hold on;
plot(X2(:, 1), X2(:, 2), 'm.');

% Calculate the Gaussian response for every value in the grid.
z1_ = GaussianNormalDist(gridX, mu1_opt1, sigma1_opt1);
z2_ = GaussianNormalDist(gridX, mu2_opt1, sigma2_opt1);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1_ = reshape(z1_, gridSize, gridSize);
Z2_ = reshape(z2_, gridSize, gridSize);

% Plot the contour lines (5) to show the pdf over the data.
contour(u, u, Z1_, 5);
contour(u, u, Z2_, 5);
daspect([1 1 1])

plot(mu1(1),mu1(2),'k.','MarkerSize',20)    %plot original center X1
plot(mu2(1),mu2(2),'k.','MarkerSize',20)    %plot original center X2
plot(mu1_opt1(1),mu1_opt1(2),'k*')  %plot new center X1
plot(mu2_opt1(1),mu2_opt1(2),'k*')  %plot new center X2
title('NMS Contours');

% QN Optimtool Estimates
subplot(2,3,6);
hold off;
plot(X1(:, 1), X1(:, 2), 'g.');
hold on;
plot(X2(:, 1), X2(:, 2), 'm.');

% Calculate the Gaussian response for every value in the grid.
z1_ = GaussianNormalDist(gridX, mu1_opt2, sigma1_opt2);
z2_ = GaussianNormalDist(gridX, mu2_opt2, sigma2_opt2);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1_ = reshape(z1_, gridSize, gridSize);
Z2_ = reshape(z2_, gridSize, gridSize);

% Plot the contour lines (5) to show the pdf over the data.
contour(u, u, Z1_, 5);
contour(u, u, Z2_, 5);
daspect([1 1 1])

plot(mu1(1),mu1(2),'k.','MarkerSize',20)    %plot original center X1
plot(mu2(1),mu2(2),'k.','MarkerSize',20)    %plot original center X2
plot(mu1_opt2(1),mu1_opt2(2),'k*')  %plot new center X1
plot(mu2_opt2(1),mu2_opt2(2),'k*')  %plot new center X2
title('QN Contours');

%% Clear extra variables
clear i j k h m n 
clear class1 class2 dim1 dim2 indeces
clear maxiter iter tol numsamples
clear A B C Xm W weights val Xclass X1 X2 x0 x0_ X
clear gridSize gridX maxgrid 
clear options options2 optXval optXval2 optOut optOut2 
clear optFval optFval2 optFlag optFlag2
clear Z Z1 Z2 z1 z2 Z1_ Z2_ z1_ z2_ mu1g mu2g u
clear sigma_k sigma R1 R2 pdf pdf_w  prevMu mu phi

%% Time entire script runtime
toc