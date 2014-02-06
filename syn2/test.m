close all; clear all;

%seed = 3;
seed = 1;
rand('seed',seed); randn('seed',seed);
M = 15;
set(gcf,'defaultlinelinewidth',1.5);

load('gp.mat');

N = 200;
Ns = 500;
D = 1;
% for ind = 1:10
% x = linspace(0,3,N)';
% %x = rand(N,1)*3;
% %x = rand(N,1)*2*pi;
% y = x.*sin(x.^3) + randn(N, 1)*0.5;
% %y = sin(2*pi*2*x);
% %y(x < pi) = sin(2*pi*x(x<pi));
% xtest = linspace(0, 3, Ns)';
% %xtest = linspace(0, 2*pi, Ns)';
% ytest = xtest.*sin(xtest.^3);
% save(strcat('data/syn2_', int2str(ind), '.mat'), 'x', 'y', 'xtest', 'ytest');
% end



for ind = 1:10
load(strcat('syn2_', int2str(ind), '.mat'));
%%
% initialize hyperparameters sensibly (see spgp_lik for how
% the hyperparameters are encoded)
model.logSigma = log(var(y,1)/4); % log noise
model.logEta = 2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
model.logA0 = log(var(y,1)); % log size 
model.logA1 = log(1); % log size 
model.logA2 = log(0.1); % log size 

trained_model = EigenGPNS_train(model, x, y, M, 100);
[mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);

nmse_ns(ind) = mean((mu-mu_gp{ind}).^2)/mean((mean(mu)-mu_gp{ind}).^2);
kl_ns(ind) = mean(s2_gp{ind}./s2 + (mu-mu_gp{ind}).^2./s2-1-log(s2_gp{ind}./s2))/2;

clf
hold on
plot(xtest, ytest, '-', 'Color', [0 .5 0]);
plot(x,y,'.m', 'MarkerSize', 15) % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(trained_model.B,-2.75*ones(size(trained_model.B)),'k+','markersize',20)
hold off
axis([-0 3 -4 5])
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
saveas(gcf, strcat('fig/syn_EigenGP_kerB_ns_', int2str(ind),'.pdf'), 'pdf');

end


fprintf('avarage nmse: %f\nstd err: %f\n', mean(nmse_ns), std(nmse_ns)/sqrt(10));
%fprintf('avarage kl: %f\nstd err: %f\n', mean(kl_ns),
%std(kl_ns)/sqrt(numTest));