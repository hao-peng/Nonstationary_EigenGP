function test
load('synthetic.mat');

[N,D] = size(x);

M = 10; % number of pseudo-inputs

seed = 1;
rand('seed',seed); randn('seed',seed);
model.logSigma = log(var(y,1)/4);
model.logEta = -2*log((max(x)-min(x))'/2);
model.logA0 = log(var(y,1));
model.logA1 = 0.1;
model.logA2 = 0.1;
model.B = x(randsample(N, M), :);

param = EigenGP_model2param(model, D, M);

EigenGP_negLogLik(param, x, y, M);
end