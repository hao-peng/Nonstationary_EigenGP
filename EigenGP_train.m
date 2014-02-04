% Train EigenGP model
% Parameters:
% initModel - initial values for parameters of EigenGP
%    initModel requires logSigma, logEta, logA0, logA1 and logA2
%    if B is not initialized in initModel, we use kmeans+ to initialize it.
% trainX - training data
%    N by D matrix. Each row is a data point.
% trainY - tarining labels
%    N by 1 matrix. Each row is a label.
% M - number of basis used
% nIter - maximum number of iterations for optimization

function model = EigenGP_train(initModel, trainX, trainY, M, nIter)
% Get the dimension of input
D = size(trainX, 2);

% Initialize B if it is not given
if ~isfield(initModel, 'B')
    %use kmeans with both x and y
    [IDX, B] = fkmeans(trainX', M);
    initModel.B = B;
    clear IDX;
end
param = EigenGP_model2param(initModel, D, M);

% Train EigenGP by minimizing the log likelihood.
[new_param] = minimize(param, @(param) EigenGP_negLogLik(param, trainX, trainY, M), nIter);
model = EigenGP_param2model(new_param, D, M);

end