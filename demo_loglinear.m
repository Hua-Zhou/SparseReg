%% simulate data (non-orthogonal design)

clear;
n = 100;
p = 500;

X = randn(n,p);   % design matrix
X = [ones(size(X,1),1) X];
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
b = zeros(p+1,1);
b(2:11) = 1;
inner = X*b;   % linear parts
poissmean = exp(inner);
y = poissrnd(poissmean);
wt = ones(n,1);

%% test glm_maxlambda()

c = [];
pentype = 'enet';
penparam = 1.5;
model = 'loglinear';
for j=1:size(X,2)
    maxlambda = glm_maxlambda(X(:,j),c,y,wt,pentype,penparam,model);
    display(maxlambda);
end

%% test glm_thresholding()

C = zeros(size(X,1),1);
pentype = 'enet';
penparam = 1;
model = 'loglinear';
lambda = .5;
betahat = ...
    glm_thresholding(X,C,y,wt,lambda,pentype,penparam,model);
display(betahat');

%% test glm_sparsereg()

penidx = [false; true(size(X,2)-1,1)];
pentype = 'log';
penparam = 1;
model = 'loglinear';
x0 = [];
maxiter = [];
lambda = 1;
betahat = ...
    glm_sparsereg(X,y,wt,lambda,x0,penidx,maxiter,pentype,penparam,model);
display(betahat');

%% individual tests

maxpreds = 10;
model = 'loglinear';
pentype = 'power';
penparam = .5;
penidx = [false; true(size(X,2)-1,1)];
wt = [];
profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    glm_sparsepath(X,y,wt,penidx,maxpreds,pentype,penparam,model);
timing = toc;
profile viewer;

figure;
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([pentype ':\eta=' num2str(penparam) ', ' num2str(timing) ' secs']);

%% test glm_sparsepath

penalty = {'enet' 'enet' 'power' 'power' 'log' 'log'...
    'mcp' 'scad'};
penidx = [false; true(size(X,2)-1,1)];
eta = [1 1.5 0.5 1 0 1 1 3.7];
model = 'loglinear';
wt = [];
maxpreds = [];

% penalty = {'enet' 'enet' 'power' 'power' 'log' ...
%     'mcp' 'scad'};
% eta = [1 1.5 0.5 1 0 1 3.7];

figure;
for i=1:length(penalty)
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    glm_sparsepath(X,y,wt,penidx,maxpreds,penalty{i},eta(i),model);
timing = toc;
display(timing);
% profile viewer;

subplot(4,3,i);
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty{i} ':\eta=' num2str(eta(i)) ', ' num2str(timing) ' secs']);

% figure;
% [AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
%     rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
% xlabel('\rho');
% set(get(AX(1),'Ylabel'),'String','negative log-likelihood'); 
% set(get(AX(2),'Ylabel'),'String','number of parameters');
% title([penalty ': \eta=' num2str(eta)]);
end
text(1.2*max(rho_path),0,[model ', n=' num2str(n) ', p=' num2str(p) ', ' ...
    ' maxpreds=' num2str(maxpreds)],'FontSize',15,'HorizontalAlignment','left');

% orient landscape
% print -depsc2 ../../manuscripts/notes/testing03.eps;