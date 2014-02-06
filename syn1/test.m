close all; clear all;
load('synthetic.mat');
addpath('..');
[N,D] = size(x);

M = 5; % number of pseudo-inputs

%range = 71:208;
%range = 1:277; %[-3 9]
%range = 48:231; %[-1 7]
range = 71:208; %[0 6]
%xtest = xtest(1:277);
%xtest = xtest(71:208);
seed = 1;
rand('seed',seed); randn('seed',seed);
set(gcf,'defaultlinelinewidth',1.5);



%set(0,'DefaultSurfaceEdgeColor', [0 .5 0])
% initialize hyperparameters sensibly (see spgp_lik for how
% the hyperparameters are encoded)
%opt.cov(1:D) = -2*log((max(x)-min(x))'); % log 1/(lengthscales)^2
opt.cov(1:D) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
opt.cov(D+1) = log(var(y,1)); % log size 
opt.lik = log(var(y,1)/4); % log noise
%[dum,I] = sort(rand(N,1)); clear dum;
%I = I(1:M);
%opt.B = x(I,:);
%opt.nIter = 10;


%%
%meanp = mean(y);
%yn = y - meanp;
hyp1 = minimize(opt, @gp, 100, @infExact, [], {@covSEard}, @likGauss, x, y);
[mu s2] = gp(hyp1, @infExact, [], {@covSEard}, @likGauss, x, y, xtest);
%mu = mu + meanp;
clf
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
%plot(post.opt.B,-2.75*ones(size(post.opt.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf, 'fig/syn_fullGP.pdf', 'pdf');

mu_full = mu;
s2_full = s2;


numTest = 10;

for tid = 1:numTest
model.logSigma = opt.lik;
model.logEta = opt.cov(1:D,1)*rand(D,1);
model.logA0 = opt.cov(D+1)*rand();
model.logA1 = 0.1*rand();
model.logA2 = 0.1*rand();

trained_model = EigenGPNS_train(model, x, y, M, 50);
[mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);

nmse_ns(tid) = mean((mu(range)-mu_full(range)).^2)/mean((mean(mu(range))-mu_full(range)).^2);
kl_ns(tid) = mean(s2_full(range)./s2(range) + (mu(range)-mu_full(range)).^2./s2(range)-1-log(s2_full(range)./s2(range)))/2;
clf
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(trained_model.B,-2.75*ones(size(trained_model.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
filename = strcat('fig/syn_EigenGP_kerB_ns_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end


fprintf('avarage nmse: %f\nstd err: %f\n', mean(nmse_ns), std(nmse_ns)/sqrt(numTest));
%fprintf('avarage kl: %f\nstd err: %f\n', mean(kl_ns), std(kl_ns)/sqrt(numTest));