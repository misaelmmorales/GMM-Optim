# GMM-Optim
The expectation-maximization (EM) algorithm is a good way to find the local maximum likelihood estimate for a set of statistical parameters, namely mean and variance of a Gaussian data set. This project shows how to implement the EM algorithm for a multiclass, multidimensional Gaussian data, and further refining the MLE estimators using MATLAB's Optimization toolbox.

### Notes ###
- *__GMM.m__* is the main .m-file. The user defines the true mean, variance, and proportions of the data set, which is then randomly generated in *GaussianNormalDist.m*. The negative log-likelihood objective function is computed via *GMM_negloglik.m*. After obtaining the EM estimates, *optimtool* will use the Nelder-Mead/Simplex and BFGS Quasi-Newton to further refine the EM estimates.

<img src="https://github.com/misaelmmorales/GMM-Optim/blob/main/imgs/mv_data_hist.jpg" width="400"> <img src="https://github.com/misaelmmorales/GMM-Optim/blob/main/imgs/mv_data_contours.jpg" width="400">
