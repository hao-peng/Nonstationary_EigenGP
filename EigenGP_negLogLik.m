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
X_B = X*B';
B_B = B*B';
% Compute gram matrices
expH = exp(bsxfun(@minus,bsxfun(@minus,2*scale_cols(X,eta)*B',X2*eta),(B2*eta)'));
Kxb = a0*expH+a1*(X_B)+a2;
expF = exp(bsxfun(@minus,bsxfun(@minus,2*scale_cols(B,eta)*B',B2*eta),(B2*eta)'));
Kbb = a0*expF+a1*(B_B)+a2; 
% Define Q = Kbb + 1/sigma2 * Kbx *Kxb
Q = Kbb+(Kxb'*Kxb)/sigma2;
% Cholesky factorization for stable computation
cholQ = chol(Q,'lower');
cholKbb = chol(Kbb,'lower');
% Ohter ocmmonly used terms
lowerOpt.LT = true; upperOpt.LT = true; upperOpt.TRANSA = true;
invCholQ_Kbx_invSigma2 = linsolve(cholQ,Kxb'/sigma2,lowerOpt);
invCholQ_Kbx_invSigma2_t = invCholQ_Kbx_invSigma2*t;
diagInvCN = 1/sigma2-sum(invCholQ_Kbx_invSigma2.^2, 1)';
invCN_t = t/sigma2-invCholQ_Kbx_invSigma2'*invCholQ_Kbx_invSigma2_t;

% compute negative log likelihood function f = (ln|CN|+t'*CN*t+ln(2*pi))/2
f = sum(log(diag(cholQ)))-sum(log(diag(cholKbb)))+(log(sigma2)*N+...
    t'*t/sigma2-invCholQ_Kbx_invSigma2_t'*invCholQ_Kbx_invSigma2_t...
    +N*log(2*pi))/2;

%f = sum(log(diag(cholQ)))-sum(log(diag(cholKbb)))+(log(sigma2)*N)/2;

% for debug
%CN = Kxb*(Kbb\Kxb')+sigma2*diag(ones(N,1));
%t_invCN_Kxb_invKbb = (CN\Kxb)/Kbb;


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
% May use later
invKbb_Kbx_invCN = linsolve(cholQ,invCholQ_Kbx_invSigma2,upperOpt);
invKbb_Kbx_invCN_Kxb_invKbb = linsolve(cholKbb, linsolve(cholKbb, Kxb'*invKbb_Kbx_invCN',lowerOpt),upperOpt)';
%invKbb_Kbx_invCN_Kxb_invKbb = inv(Kbb) - inv(Q)
invKbb_Kbx_invCN_t = invKbb_Kbx_invCN*t;
invKbb_Kbx_invCN_t_t_invCN = invKbb_Kbx_invCN_t*invCN_t';
invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb = invKbb_Kbx_invCN_t*invKbb_Kbx_invCN_t';

% compute dlogSigma
dlogSigma = sigma2*(sum(diagInvCN)-invCN_t'*invCN_t);

% compute dlogA0
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN.*expH'))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb.*expF'));
%part1 = 2*sum(sum(invKbb_Kbx_invCN.*expH'))-trace(invKbb_Kbx_invCN_Kxb_invKbb_expF);
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN.*expH'))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb.*expF'));
dlogA0 = a0*(part1-part2)/2;

% compute dlogA1
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN.*X_B'))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb.*B_B));
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN.*X_B'))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb.*B_B'));
dlogA1 = a1*(part1-part2)/2;

% compute dlogA2
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb));
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb));
dlogA2 = a1*(part1-part2)/2;


% combine all gradients in a vector
df = [dlogSigma; dlogEta; dlogA0; dlogA1; dlogA2; reshape(dB,D*M,1)];
end
