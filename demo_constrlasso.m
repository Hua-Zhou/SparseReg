%% lasso with a linear equality constraint

% set seed
clear;
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% dimension
n = 100;
p = 20;

% truth with sum constraint sum(b)=0
beta = zeros(p,1);
beta(1:round(p/4)) = 0;
beta(round(p/4)+1:round(p/2)) = 1;
beta(round(p/2)+1:round(3*p/4)) = 0;
beta(round(3*p/4)+1:end) = -1;

% generate data
X = randn(n,p);
y = X*beta + randn(n,1);

% penalty parameter
lambda = 168.3;
Aeq = ones(1,p);
beq = 0;

% fit using quadprog
tic;
bhat_qp = lsq_constrsparsereg(X,y,lambda,...
    'method','qp','qp_solver','matlab','Aeq', Aeq, 'beq', beq);
toc;

% fit constrained lasso using GUROBI
tic;
bhat_gurobi = lsq_constrsparsereg(X,y,lambda,...
    'method','qp','qp_solver','GUROBI','Aeq', Aeq, 'beq', beq);
toc;

% fit constrained lasso using ADMM (quadprog for projection subproblem)
tic;
[bhat_admm_matlab,stats] = lsq_constrsparsereg(X,y,lambda,...
    'method','admm','qp_solver','matlab','Aeq', Aeq, 'beq', beq);
toc;
display(stats.ADMM_iters);

% fit constrained lasso using ADMM (GUROBI for projection subproblem)
tic;
[bhat_admm_gurobi,stats] = lsq_constrsparsereg(X,y,lambda,...
    'method','admm','qp_solver','GUROBI','Aeq', Aeq, 'beq', beq);
toc;
display(stats.ADMM_iters);

% fit constrained lasso using ADMM (function handle for projection subproblem)
tic;
%profile on;
[bhat_admm_fh,stats] = lsq_constrsparsereg(X,y,lambda,...
    'method','admm','projC', @(x) x-mean(x));
%profile viewer;
toc;
display(stats.ADMM_iters);

% plot solutions
figure; hold on;
plot(bhat_qp,'rs');
plot(bhat_gurobi,'-b+');
plot(bhat_admm_matlab,'go');
plot(bhat_admm_gurobi,'m*');
plot(bhat_admm_fh,'kv');
legend('quadpro', 'GUROBI', 'ADMM (quadprog)', 'ADMM (GUROBI)', 'ADMM (FH)');

%% lasso constrained on a box

% set seed
clear;
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% dimension
n = 1000;
p = 1000;

% truth with constraint b>=0 and b<=1
beta = zeros(p,1);
beta(1:round(p/4)) = 0;
beta(round(p/4)+1:round(p/2)) = 1;
beta(round(p/2)+1:round(3*p/4)) = 0;
beta(round(3*p/4)+1:end) = 1;

% generate data
X = randn(n,p);
y = X*beta + randn(n,1);

% penalty parameter
lambda = 100;

% fit constrained lasso using quadprog
A = [- eye(p); eye(p)];
b = [zeros(p,1); ones(p,1)];
tic;
bhat_qp = lsq_constrsparsereg(X,y,lambda,...
    'method','qp','qp_solver','matlab','A',A,'b',b);
toc;

% fit constrained lasso using GUROBI
tic;
bhat_gurobi = lsq_constrsparsereg(X,y,lambda,...
    'method','qp','qp_solver','GUROBI','A',A,'b',b);
toc;

% % fit constrained lasso using ADMM (quadprog for projection subproblem)
% tic;
% [bhat_admm_matlab,stats] = lsq_constrsparsereg(X,y,lambda,...
%     'method','admm','qp_solver','matlab',...
%     'admmVaryScale',true,'A',A,'b',b);
% toc;
% display(stats.ADMM_iters);
% 
% % fit constrained lasso using ADMM (GUROBI for projection subproblem)
% tic;
% [bhat_admm_gurobi,stats] = lsq_constrsparsereg(X,y,lambda,...
%     'method','admm','qp_solver','GUROBI',...
%     'admmVaryScale',true,'A',A,'b',b);
% toc;
% display(stats.ADMM_iters);

% fit constrained lasso using ADMM (func handle for projection subproblem)
tic;
[bhat_admm_fh,stats] = lsq_constrsparsereg(X,y,lambda,...
    'method','admm','projC', @(x) min(max(x,0),1));
toc;
display(stats.ADMM_iters);

% plot solutions
figure; hold on;
plot(bhat_qp,'rs');
plot(bhat_gurobi,'-b+');
% plot(bhat_admm_matlab,'go');
% plot(bhat_admm_gurobi,'m*');
plot(bhat_admm_fh,'kv');
legend('quadpro', 'GUROBI', 'ADMM (quadprog)', 'ADMM (GUROBI)', 'ADMM (FH)');

%% generalized lasso via constrained lasso
% Let's solve a sparse fused lasso problem by transforming to constr. lasso

% set seed
clear;
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% dimension
n = 500;
p = 100;

% truth with piecewise linearity
beta = zeros(p,1);
beta(1:round(p/4)) = 0;
beta(round(p/4)+1:round(p/2)) = 1;
beta(round(p/2)+1:round(3*p/4)) = 0;
beta(round(3*p/4)+1:end) = 1.5;

% generate data
X = randn(n,p);
y = X*beta + randn(n,1);

% penalty parameter and penalty matrix
lambda = 10;
D = [eye(p-1) zeros(p-1,1)] - [zeros(p-1,1) eye(p-1)];
D(p:2*p-1, :) = eye(p);
m = size(D,1);

% transform to constrained lasso
[U,s,V] = svd(D);
s = diag(s);
r = nnz(s > eps(s(1))*max(size(D)));
V1 = V(:,1:r); V2 = V(:,r+1:end);
U1 = U(:,1:r); U2 = U(:,r+1:end);
s1 = s(1:r);

pinvD = V1*bsxfun(@times, U1', 1./s1);
XpinvD = X*pinvD;

% solve constrained lasso
tic;
[alphahat_admm_fh,stats] = lsq_constrsparsereg(XpinvD,y,lambda,...
    'method','admm','projC', @(x) U1*(U1'*x),...
    'admmScale',1/n);
toc;
display(stats.ADMM_iters);
% back to original parameterization
betahat_fh = pinvD*alphahat_admm_fh;

% solve constrained lasso
tic;
[alphahat_admm_gurobi,stats] = lsq_constrsparsereg(XpinvD,y,lambda,...
    'method','admm','qp_solver', 'gurobi', 'Aeq', U2', 'beq',zeros(m-r,1),...
    'admmScale',1/n);
toc;
display(stats.ADMM_iters);
betahat_gurobi = pinvD*alphahat_admm_gurobi;

% plot solutions
figure; hold on;
plot(betahat_fh,'rs');
plot(betahat_gurobi,'-b+');
legend('ADMM (FH)', 'ADMM (GUROBI)');
