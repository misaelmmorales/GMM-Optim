function Gausspdf = GaussianNormalDist(X, mu, Sigma) 
% X     - data frame matrix
% mu    - row vector for the means
% Sigma - square covariance matrix

% Vector length. This is the number of dimensions
n = size(X, 2);

% Simplify the formula by using an element-wise operation (for @minus)
% We subtract the mean (mu) from every data point (X).
meanDiff = bsxfun(@minus, X, mu);

% Calculate the Multivariate Normal/Gaussian Distribution .
Gausspdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));

% end function
end

