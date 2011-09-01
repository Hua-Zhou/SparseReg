%% Test lsq_sparsereg function

clear;
n = 100;
p = 10000;

X = randn(n,p-1);   % design matrix
X = [ones(size(X,1),1) X];
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
b = zeros(p,1);
b(1:10) = 1;
y = normrnd(X*b,1,n,1);   % response vector
wt = ones(n,1);
penidx = [false; true(size(X,2)-1,1)];
sum_x_squares = sum(bsxfun(@times, wt, X.^2),1)';
x0 = zeros(size(X,2),1);

tic;
betahat = lsq_sparsereg(X,y,wt,.01,x0,sum_x_squares,penidx,[],'power',1);
toc;
display(betahat(1:20)');

%% Test penalty_function

[pen,d1pen,d2pen,dpendlambda] = penalty_function((1:5),1,'enet',1);
display(pen);
display(d1pen);
display(d2pen);
display(dpendlambda);

%% test lsq_thresholding function

a = 1:11;
b = -5:5;
lambda = 1;
[xmin] = lsq_thresholding(a,b,lambda,'mcp',1);
display(xmin);

%% test lsq_thresholding function

a = 1;
b = 1;
[maxlambda] = lsq_maxlambda(a,b,'power',.5);
display(maxlambda);

%% simulate (non-orthogonal design)

clear;
n = 100;
p = 5000;
maxpreds = 50;

X = randn(n,p);   % design matrix
X = [ones(size(X,1),1) X];
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
b = zeros(p+1,1);
b(2:6) = 1;
y = normrnd(X*b,1,n,1);   % response vector
wt = ones(n,1);
penidx = [false; true(size(X,2)-1,1)];

%% test lsq_sparsepath

penalty = 'scad';
eta = 3.7;
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty,eta);
toc;
% profile viewer;

figure; hold on;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title([penalty ': \eta=' num2str(eta)]);
% print -depsc2 ../../manuscripts/notes/toy_p10_solpath_DP.eps;

figure;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters') 
title([penalty ': \eta=' num2str(eta)]);
