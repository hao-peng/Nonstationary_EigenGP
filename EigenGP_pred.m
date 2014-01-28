% Use a EigenGP model to predict 
% Parameters:
% model - initial values for parameters of EigenGP
%    initModel requires logSigma, logEta, logA0, logA1 and logA2
%    if B is not initialized in initModel, we use kmeans+ to initialize it.
% testX - test data
%    N by D matrix. Each row is a data point.
% M - number of basis used

function pred = EigenGP_pred(model, testX, M)

end