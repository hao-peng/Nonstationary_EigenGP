function test
%load('synthetic.mat');
load('syn2_1.mat');
[N,D] = size(x);

M = 20; % number of pseudo-inputs

seed = 0;
rand('seed',seed); randn('seed',seed);
model.logSigma = log(var(y,1)/4);
model.logEta = 2*log((max(x)-min(x))'*2);
model.logA0 = log(var(y,1));
%model.logA0 = log(0.1);
model.logA1 = log(0.1);
model.logA2 = log(0.1);
%model.B = linspace(min(x), max(x), M);
%model.B = x(randsample(N, M), :);

trained_model = EigenGP_train(model, x, y, M, 100);


param = EigenGP_model2param(trained_model, D, M);

EigenGP_negLogLik(param, x, y, M);

% index =15;
% range = param(index)-1:0.01:param(index)+1;
% for i = 1:size(range,2);
%     param(index) = range(i);
%     [f(i), g] = EigenGP_negLogLik(param, x, y, M);
%     df(i) = g(index);
% end
% for i = 2:size(range,2)-1;
%     fdf(i-1) = (f(i+1)-f(i-1))/(range(i+1)-range(i-1));
% end
% subplot(2,1,1);
% plot(range, f);
% subplot(2,1,2);
% plot(range, df);
% hold on;
% plot(range(1,2:size(range,2)-1), fdf, 'r');
% hold off;

[mu s2] = EigenGP_pred(trained_model, x, y, xtest);

set(gcf,'defaultlinelinewidth',1.3);
plot(x,y,'.m','MarkerSize', 12);
hold on;
plot(xtest, mu);
plot(xtest, mu+2*sqrt(s2), 'r');
plot(xtest, mu-2*sqrt(s2), 'r');
plot(trained_model.B,(min(y)-0.5)*ones(M,1), '+g', 'MarkerSize', 12);
hold off;
end