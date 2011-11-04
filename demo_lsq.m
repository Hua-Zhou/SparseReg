%% Sparse Linear Regression
% A demonstration of sparse linear regression using SparseReg 
% toolbox. Sparsity is in the general sense: variable selection, 
% fused sparse regression (total variation regularization), polynomial trend 
% filtering, and others. Various penalties are implemented:
% elestic net (enet), power family (bridge regression), log penalty, SCAD, 
% and MCP.

%% Sparse linear regression (n>p)
% Simulate a sample data set (n=500, p=100)
clear;
n = 500;
p = 100;
X = randn(n,p);             % generate a random design matrix
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize predictors
X = [ones(size(X,1),1) X];  % add intercept
b = zeros(p+1,1);           % true signal: 
b(2:6) = 3;                 % first 5 predictors are 3
b(7:11) = -3;               % next 5 predictors are -3
y = X*b+randn(n,1);         % response vector

%%
% Sparse regression at a fixed tuning parameter value
penalty = 'enet';           % set penalty function to lasso
penparam = 1;
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
lambdastart = ...           % find the maximum tuning parameter to start
    max(lsq_maxlambda(sum(X(:,penidx).^2),-y'*X(:,penidx),penalty,penparam));
display(lambdastart);

lambda = 0.9*lambdastart;   % tuning parameter value
maxiter = [];               % use default maximum number of iterations
sum_x_squares = [];         % compute predictor norms on the fly
wt = [];                    % use default observation weights (1)
x0 = [];                    % use default start value (0)
betahat = ...               % sparse regression
    lsq_sparsereg(X,y,wt,lambda,x0,sum_x_squares,penidx,maxiter,penalty,penparam);
figure;                     % plot penalized estimate
bar(1:length(betahat),betahat);
xlabel('j');
ylabel('\beta_j');
xlim([0,length(betahat)+1]);
title([penalty '(' num2str(penparam) '), \lambda=' num2str(lambda,2)]);

lambda = 0.5*lambdastart;   % try a smaller tuning parameter value
betahat = ...               % sparse regression
    lsq_sparsereg(X,y,wt,lambda,x0,sum_x_squares,penidx,maxiter,penalty,penparam);
figure;                     % plot penalized estimate
bar(1:length(betahat),betahat);
xlabel('j');
ylabel('\beta_j');
xlim([0,length(betahat)+1]);
title([penalty '(' num2str(penparam) '), \lambda=' num2str(lambda,2)]);

%% 
% Solution path for lasso
maxpreds = [];              % try to obtain the whole solution path
penalty = 'enet';           % set penalty function
penparam = 1;
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
wt = ones(n,1);             % equal weights for all data points
tic;
[rho_path,beta_path] = ...  % compute solution path
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,penparam);
timing = toc;

figure;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);

%% 
% Solution path for power (0.5)
penalty = 'power';          % set penalty function to power
penparam = 0.5;
tic;
[rho_path,beta_path] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,penparam);
timing = toc;

figure;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);

%% 
% Compare solution paths from different penalties
maxpreds = [];              % try to obtain the whole solution paths
penalty = {'enet' 'enet' 'enet' 'power' 'power' 'log' 'log' 'mcp' 'scad'};
penparam = [1 1.5 2 0.5 1 0 1 1 3.7];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
wt = ones(n,1);             % equal weights for all data points

figure
for i=1:length(penalty)
    tic;
    [rho_path,beta_path] = ...
        lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty{i},penparam(i));
    timing = toc;
    subplot(3,3,i);
    plot(rho_path,beta_path);
    if (i==8)
        xlabel('\rho');
    end
    if (i==4) 
        ylabel('\beta(\rho)');
    end
    xlim([min(rho_path),max(rho_path)]);
    title([penalty{i} '(' num2str(penparam(i)) '), ' num2str(timing,1) 's']);
end

%% Fused linear regression
% Fused lasso (fusing the first 10 predictors)
D = zeros(9,size(X,2));     % regularization matrix for fusing first 10 preds
D(10:10:90) = 1;            
D(19:10:99) = -1;
display(D(1:9,1:11));
penalty = 'enet';           % set penalty function to lasso
penparam = 1;
wt = [];                    % equal weights for all observations
tic;
[rho_path, beta_path] = lsq_regpath(X,y,wt,D,penalty,penparam);
timing = toc;

figure;
plot(rho_path,beta_path(2:11,:));
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);

%%
% Same fusion problem, but with power, log, MCP, and SCAD penalty
penalty = {'power' 'log' 'mcp' 'scad'};
penparam = [0.5 1 1 3.7];
for i=1:length(penalty)
    tic;
    [rho_path, beta_path] = lsq_regpath(X,y,wt,D,penalty{i},penparam(i));
    timing = toc;
    subplot(2,2,i);
    plot(rho_path,beta_path(2:11,:));
    xlim([min(rho_path),max(rho_path)]);
    title([penalty{i} '(' num2str(penparam(i)) '), ' num2str(timing,1) 's']);
end

%% Sparse linear regression (n<p)
% Simulate another sample data set (n=100, p=500)
clear;
n = 100;
p = 500;
X = randn(n,p);             % generate a random design matrix
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize predictors
X = [ones(size(X,1),1) X];  % add intercept
b = zeros(p+1,1);           % true signal
b(2:6) = 3;                 % first 5 predictors are 3
b(7:11) = -3;               % next 5 predictors are -3
y = X*b+randn(n,1);         % response vector

%% 
% Solution path for lasso
maxpreds = 50;              % run solution path until 50 predictors are in
penalty = 'enet';           % set penalty function
penparam = 1;
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
wt = ones(n,1);             % equal weights for all data points
tic;
[rho_path,beta_path] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,penparam);
timing = toc;

figure;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);

%% 
% Solution path for power (0.5)
penalty = 'power';          % set penalty function to power
penparam = 0.5;
tic;
[rho_path,beta_path] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,penparam);
timing = toc;

figure;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);