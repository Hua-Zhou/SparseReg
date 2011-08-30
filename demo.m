%% simulate (non-orthogonal design)

clear;
n = 100;
p = 10000;
maxpreds = [];

X = randn(n,p);   % design matrix
X = [ones(size(X,1),1) X];
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
b = zeros(p+1,1);
b(1:10) = 1;
y = normrnd(X*b,1,n,1);   % response vector
wt = ones(n,1);
% penidx = [false; true(size(X,2)-1,1)];
penidx = [true(size(X,2),1)];
sum_x_squares = sum(bsxfun(@times, wt, X.^2),1)';
x0 = zeros(size(X,2),1);

%% mex function
tic;
betahat = lsqsparse(x0,X,y,wt,0.1,sum_x_squares,penidx,50,'POWER',0.5);
toc;
display(betahat(1:20)');

%% sparse regression - double pareto penalty with parameter eta
eta = .5;    % parameter for double pareto
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = lsq_sparsereg(@penfun_pareto, ...
    @univoptm_pareto,X,y,wt,penidx,maxpreds,eta);
toc;
% profile viewer;

figure; hold on;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title(['DP: \eta=' num2str(eta)]);
% print -depsc2 ../../manuscripts/notes/toy_p10_solpath_DP.eps;

figure;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters') 
title(['DP: \eta=' num2str(eta)]);

%% sparse regression - double pareto penalty (continuous version)

% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = lsq_sparsereg(@penfun_pareto, ...
    @univoptm_pareto,X,y,wt,penidx,maxpreds,[]);
toc;
% profile viewer;

figure; hold on;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title('DP: \eta=\rho^{1/2}');
print -depsc2 ../../manuscripts/notes/toy_p10_solpath_CDP.eps;

figure;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters') 
title('DP: \eta=\rho^{1/2}');

%% sparse regression - elastic net

lambda = 1; % parameter for elastic net (0,2]
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = lsq_sparsereg(@penfun_enet, ...
    @univoptm_enet,X,y,wt,penidx,maxpreds,lambda);
toc;
% profile viewer;

figure; hold on;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title(['Enet: \lambda=' num2str(lambda)]);
print -depsc2 ../../manuscripts/notes/toy_p10_solpath_lasso.eps;

figure;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters') 
title(['Enet: \lambda=' num2str(lambda)]);

%% sparse regression - bridge

lambda = .8; % parameter for bridge (0,2]
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = lsq_sparsereg(@penfun_bridge,...
    @univoptm_bridge,X,y,wt,penidx,maxpreds,lambda);
toc;
% profile viewer;

figure; hold on;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title(['Bridge: \lambda=' num2str(lambda)]);
print -depsc2 ../../manuscripts/notes/toy_p10_solpath_power.eps;

figure; hold on;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters')
title(['Bridge: \lambda=' num2str(lambda)]);

%% sparse regression - SCAD

eta = 3.7; % parameter for SCAD >2
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = lsq_sparsereg(@penfun_scad,...
    @univoptm_scad,X,y,wt,penidx,maxpreds,eta);
toc;
% profile viewer;

figure; hold on;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
title(['SCAD: \eta=' num2str(eta)]);

figure; hold on;
[AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
    rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
xlabel('\rho');
set(get(AX(1),'Ylabel'),'String','negative log-likelihood') 
set(get(AX(2),'Ylabel'),'String','number of parameters')
title(['SCAD: \eta=' num2str(eta)]);