%% lasso with a linear equality constraint

% set seed
clear;
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% dimension
n = 1000;
p = 200;

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

%% obtain solution path by path following
tic;
[rhopath,betapath] ...
    = lsq_classopath(X,y,[],[],Aeq,beq,'qp_solver','gurobi');
timing_path = toc;

% plot solutions
figure; hold on;
set(gca,'FontSize', 20);
plot(rhopath, betapath');
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rhopath) max(rhopath)*1.05]);
title(['path following algorithm:' num2str(timing_path) ' s']);

%% obtain solution path by GUROBI optimization at grid

betapath_gurobi = zeros(size(betapath));
tic;
for k = 1:length(rhopath)
    display(k);
    [betapath_gurobi(:,k)] ...
    = lsq_constrsparsereg(X,y,rhopath(k),...
    'method','qp','qp_solver','matlab','Aeq', Aeq, 'beq', beq);
end
timing_gurobi = toc;

% plot solutions
figure; hold on;
set(gca,'FontSize', 20);
plot(rhopath, betapath_gurobi');
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rhopath) max(rhopath)*1.05]);
title(['Gurobi on grid:' num2str(timing_gurobi) ' s']);

%% obtain solution path by ADMM optimization at grid

betapath_admm = zeros(size(betapath));
tic;
for k = 1:length(rhopath)
    display(k);
    
    if k==1
        x0 = zeros(p,1);
    else
        x0 = betapath(:,k-1);
    end
    [betapath_admm(:,k)] ...
        = lsq_constrsparsereg(X,y,rhopath(k),...
        'method','admm','projC', @(x) x-mean(x),'x0',x0);
end
timing_admm = toc;

% plot solutions
figure; hold on;
set(gca,'FontSize', 20);
plot(rhopath, betapath_admm');
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rhopath) max(rhopath)*1.05]);
title(['ADMM on grid:' num2str(timing_admm) ' s']);
