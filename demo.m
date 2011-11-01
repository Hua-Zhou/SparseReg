%% SparseReg Toolbox User's Guide
% Demo for using the functions in the toolbox

%% Sparse linear regression
% simulate data (n=1000, p=100)
clear;
n = 500;
p = 100;
X = randn(n,p);   % design matrix
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
X = [ones(size(X,1),1) X];  % add intercept
b = zeros(p+1,1);   % true signal: first ten predictors are 1
b(2:11) = 1;
y = normrnd(X*b,1,n,1);   % response vector

%% 
% solution path for lasso

maxpreds = [];  % try to obtain the whole solution path
penalty = 'power';
penparam = .5;
penidx = [false; true(size(X,2)-1,1)];
wt = ones(n,1);
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,penparam);
timing = toc;

figure;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty ':\eta=' num2str(penparam) ', ' num2str(timing,2) ' secs']);

%% 
% compare solution paths from different penalteis

penalty = {'enet' 'enet' 'enet' 'power' 'power' 'log' 'log'...
    'mcp' 'scad'};
eta = [1 1.5 2 0.5 1 0 1 1 3.7];

figure
for i=1:length(penalty)
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty{i},eta(i));
timing = toc;
subplot(4,3,i);
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty{i} ':\eta=' num2str(eta(i)) ', ' num2str(timing,2) ' secs']);
end