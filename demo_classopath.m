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

% equality constraint
Aeq = ones(1,p);
beq = 0;

% generate data
X = randn(n,p);
y = X*beta + randn(n,1);

% obtain solution path
tic;
[rhopath,betapath,dualpathEq,dualpathIneq] ...
    = lsq_classopath(X,y,[],[],Aeq,beq,'qp_solver','GUROBI');
toc;


% plot solutions
%figure; hold on;