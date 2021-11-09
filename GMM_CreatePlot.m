% MISAEL MORALES  -  MATH 7993 Numerical Optimization  -  PROJECT
% ------------------------------------------------------------------------
% OPTIMAL PREDICTION AND CLUSTERING OF GAUSSIAN MIXTURE MODELS
% ------------------------------------------------------------------------
% This is a MTALAB script subroutine to create different plots for the 
% Gaussian Mixture Model. 
% It is based on the GMM.m MATLAB script.
% 
% Misael Morales - MATH 7993 - July 2020
%
% ------------------------------------------------------------------------
%% Histogram for Original Data
figure(1);
hold off
histogram(X);
hold on
title('Histogram of Multivariate Normal Data')
xlabel('X'); ylabel('Frequency');
%savefig('GMM_histogram')
% ------------------------------------------------------------------------
%% Scatterplot and Contours for Original Data

% Define parameters for grid density and dimensions.
gridSize = 100;
 maxgrid = max(max(mu1,mu2)) + max(max(max(sigma1,sigma2)));
       u = linspace(-maxgrid, maxgrid, gridSize);

% Display a scatter plot of the data from the two distributions.
% Save space as a subplot to plot side-by-side with EM-generated
% distribution estimates.
figure(2)
subplot(1,2,1);
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

title('Original Data + PDFs');

%% Scatterplot and Contours for EM-estimated Data
%
% Display a scatter plot of the data from the two distributions.
subplot(1,2,2);
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
title('Original Data + EM Contours');
%savefig('GMM_orig_EM_data_contour')

% Create a 3D Surface Plot for the pdf response function
hold off
figure(3)
surf([Z1_;Z2_])
hold on
title('Multivariate PDF for Generated Data')
%savefig('GMM_surface')

%%% end GMM_CreatePlot