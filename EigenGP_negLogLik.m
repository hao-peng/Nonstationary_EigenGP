% Compute the negative log likelihood and its derivative respect to eacch
% model paramter. We use an ARD kernel plus a linear kernel:
% k(x,y) = a0*exp(-(x-y)'*diag(eta)*(x-y))+a1*x'*y+a2
% paramters:
% param - current parameters for the model
%       element 1: log(sigma)
%       element 2 - (1+D): log(eta)
%       element (D+2): log(a0)
%       element (D+3): log(a1)
%       element (D+4): log(a2)
%       element (D+5) - (D+4+D*M): B (reshaped B matrix as a vector)
% X - input data point
%     N by D matrix, where each row is a data point
% t - labels
%     N by 1 vector
% M - number of basis point used

function [f df] = EigenGP_negLogLik(param, X, t, M)
[N D] = size(X);
% load parameters
sigma2 = exp(2*param(1));
eta = exp(param(2:1+D));
a0 = exp(param(2+D));
a1 = exp(param(3+D));
a2 = exp(param(4+D));
B = reshape(param(5+D:D*M+D+4), M, D);
% Some commonly used terms
X2 = X.*X;
B2 = B.*B;
% Compute gram matrices
expH = exp(bsxfun(@minus,bsxfun(@minus,2*scale_cols(X,eta)*B',X2*eta),(B2*eta)'));
Kxb = a0*expH+a1*(X*B')+a2;
expF = exp(bsxfun(@minus,bsxfun(@minus,2*scale_cols(B,eta)*B',B2*eta),(B2*eta)'));
Kbb = a0*expF+a1*(B*B')+a2; 
% Define Q = Kbb + 1/sigma2 * Kbx *Kxb
Q = Kbb+(Kxb'*Kxb)/sigma2;
% Cholesky factorization for stable computation
cholQ = chol(Q,'lower');
cholKbb = chol(Kbb,'lower');
% Ohter ocmmonly used terms
lowerOpt.LT = true; upperOpt.LT = true; upperOpt.TRANSA = true;
invCholQ_Kbx_invSigma2_t = linsolve(cholQ,Kxb'*t/sigma2,lowerOpt);
% compute negative log likelihood function f = (ln|CN|+t'*CN*t+ln(2*pi))/2
f = sum(log(diag(cholQ)))-sum(log(diag(cholKbb)))+(log(sigma2)*N+...
    t'*t/sigma2-invCholQ_Kbx_invSigma2_t'*invCholQ_Kbx_invSigma2_t...
    +N*log(2*pi))/2;

%-----------------------
% compute gradient
%-----------------------
% initialization gradients to be zeros
dlogSigma = 0;
dlogEta = zeros(D,1);
dlogA0 = 0;
dlogA1 = 0;
dlogA2 = 0;
dB = zeros(M,D);
% compute dlogSigma
invCN_Kxb_invKbb = linsolve(cholQ,linsolve(cholQ, Kxb', lowerOpt),uppper)'/sigma2;

% combine all gradients in a vector
df = [dlogSigma; dlogEta; dlogA0; dlogA1; dlogA2; reshape(dB,D*M,1)];
end
