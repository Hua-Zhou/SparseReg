%% simulate data (non-orthogonal design)

clear;
n = 90;
p = 20000;
maxpreds = 50;

X = randn(n,p);   % design matrix
X = [ones(size(X,1),1) X];
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize
b = zeros(p+1,1);
b(2:11) = 1;
y = normrnd(X*b,1,n,1);   % response vector
wt = ones(n,1);
penidx = [false; true(size(X,2)-1,1)];

%% individual tests

profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,'enet',1);
toc;
profile viewer;

%% test lsq_sparsepath

penalty = {'enet' 'enet' 'power' 'power' 'log' 'log'...
    'mcp' 'scad'};
eta = [1 1.5 0.5 1 0 1 1 3.7];

figure;
for i=1:length(penalty)
% profile on;
tic;
[rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penalty{i},eta(i));
toc;
% profile viewer;

subplot(4,3,i);
set(gca,'FontSize',15);
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty{i} ':\eta=' num2str(eta(i)) ', ' num2str(toc) ' secs']);

% figure;
% [AX,H1,H2] = plotyy(rho_path(rho_kinks),fval_kinks,...
%     rho_path(rho_kinks),sum(beta_path(:,rho_kinks)~=0,1));
% xlabel('\rho');
% set(get(AX(1),'Ylabel'),'String','negative log-likelihood'); 
% set(get(AX(2),'Ylabel'),'String','number of parameters');
% title([penalty ': \eta=' num2str(eta)]);
end
text(1.2*max(rho_path),0,['n=' num2str(n) ', p=' num2str(p) ', ' ...
    ' maxpreds=' num2str(maxpreds)],'FontSize',15,'HorizontalAlignment','left');

orient landscape
print -depsc2 ../../manuscripts/notes/testing03.eps;