%% M&A data

clear;
load '../../datasets/MandAcleanData.mat';
X = x; clear x;
wt = ones(length(y),1);
display(names);

% discretize the covariates into nbins bins; the first bin is used as the
% reference level; effect coding is used for all category variables
XDiscretized = cell(1,7);
nbins = 10;
for j=1:7
%     [~,xout] = hist(X(:,j),nbins);
%     xbins = xout + (max(X(:,j))-min(X(:,j)))/nbins/2;
    xqt = [quantile(X(:,j),nbins-1) max(X(:,j))];
    XDisc_j = zeros(size(X,1),nbins-1);
    for i=1:size(X,1)
        c = find(X(i,j)<=xqt, 1, 'first');
        if (c>1)
            XDisc_j(i,c-1) = 1;
        else
            XDisc_j(i,:) = -1;
        end
    end
    XDiscretized{j} = XDisc_j;
end

%% cubic trend filtering on all predictors 1-7

% construct design matrix
X = ones(size(X,1),1);
for j=1:7
    X = [X XDiscretized{j}];
end

% fit regular logistic regression
b0 = glmfit(X,y,'binomial','constant','off');
[f,d1f,d2f] = glmfun(b0,X,y,wt,'binomial',[]);
display(b0');
display(d1f');

% construct penalty matrix for quadratic trend filtering
nj = cellfun(@(A) size(A,2), XDiscretized);
Aeq = zeros(0,size(X,2));
for j=1:7
    fusemat = -eye(nj(j));
    fusemat(nj(j)+1:(nj(j)+1):nj(j)^2-1) = 1;
    fusemat = fusemat^4;
    fusemat([end-3 end-2 end-1 end],:) = [];
    fusemat(end,:) = 0;
    fusemat(end,[end-2 end-1 end]) = [-1 2 -1];
    fusemat(end+1,:) = [3 0 ones(1,size(fusemat,2)-2)];
    fusemat = [zeros(size(fusemat,1),sum(nj(1:(j-1)))+1) fusemat ...
        zeros(size(fusemat,1),sum(nj(j+1:end)))];
    Aeq = [Aeq; fusemat];
end
beq = zeros(size(Aeq,1),1);
Aineq = [];
bineq = [];

% path algorithm
% profile on;
[~, rho_path, x_path, rho_kinks] = fminlin_path(@(z,v) glmfun(z,X,y,wt,'binomial',v), ...
    b0, Aeq, beq, Aineq, bineq);
% profile viewer;
% display(x');
%display(x_path);
display(rho_kinks');

%% cubic trend filtering on predictors 1-3,5-7
% monotonicity on predictor 4
% concavity on predictor 5

% construct design matrix
X = ones(size(X,1),1);
for j=1:7
    X = [X XDiscretized{j}];
end

% fit regular logistic regression
b0 = glmfit(X,y,'binomial','constant','off');
[f,d1f,d2f] = glmfun(b0,X,y,wt,'binomial',[]);
display(b0');
display(d1f');

% construct penalty matrix for quadratic trend filtering
nj = cellfun(@(A) size(A,2), XDiscretized);
Aeq = zeros(0,size(X,2));
for j=[1:3 6 7]
    fusemat = -eye(nj(j));
    fusemat(nj(j)+1:(nj(j)+1):nj(j)^2-1) = 1;
    fusemat = fusemat^4;
    fusemat([end-3 end-2 end-1 end],:) = [];
    fusemat(end,:) = 0;
    fusemat(end,[end-2 end-1 end]) = [-1 2 -1];
    fusemat(end+1,:) = [3 0 ones(1,size(fusemat,2)-2)];
    fusemat = [zeros(size(fusemat,1),sum(nj(1:(j-1)))+1) fusemat ...
        zeros(size(fusemat,1),sum(nj(j+1:end)))];
    Aeq = [Aeq; fusemat];
end
beq = zeros(size(Aeq,1),1);

j = 4;
fusemat = - eye(nj(j));
fusemat((nj(j)+1):(nj(j)+1):(nj(j)*nj(j)-1)) = 1;
fusemat(end,:) = [];
fusemat = [zeros(size(fusemat,1),sum(nj(1:(j-1)))+1) fusemat ...
    zeros(size(fusemat,1),sum(nj(j+1:end)))];
Aineq = fusemat;
j = 5;
fusemat = -eye(nj(j));
fusemat(nj(j)+1:(nj(j)+1):nj(j)^2-1) = 1;
fusemat = fusemat*fusemat;
fusemat([end-1 end],:) = [];
fusemat = [zeros(size(fusemat,1),sum(nj(1:(j-1)))+1) fusemat ...
    zeros(size(fusemat,1),sum(nj(j+1:end)))];
Aineq = [Aineq; fusemat];
bineq = zeros(size(Aineq, 1), 1);

% path algorithm
% profile on;
[~, rho_path, x_path, rho_kinks] = fminlin_path(@(z,v) glmfun(z,X,y,wt,'binomial',v), ...
    b0, Aeq, beq, Aineq, bineq);
% profile viewer;
% display(x');
%display(x_path);
display(rho_kinks');

%% plot unconstrained/constrained estimates
rhokink = size(x_path,1);
rhokink2 = size(x_path,1)-40;
load '../../datasets/MandAcleanData.mat' x;
figure;
set(gca,'FontSize', 20);
for j=1:7
    subplot(3,3,j); hold on;
    xqt = [quantile(x(:,j),nbins-1) max(x(:,j))]';
    xidx = (2+sum(nj(1:j-1))):(1+sum(nj(1:j)));
    plot(xqt,[-sum(x_path(1,xidx)) x_path(1,xidx)],'o');
    plot(xqt,[-sum(x_path(rhokink2,xidx)) x_path(rhokink2,xidx)],':');
    plot(xqt,[-sum(x_path(rhokink,xidx)) x_path(rhokink,xidx)],'-+');
    xlabel(names{j});
    xlim([min(x(:,j)) max(x(:,j))+abs(xqt(1))]);
end
legend('unconstrained estimate', ['\rho=' num2str(rho_path(rhokink2))], ...
    'constrained estimate', 'Location', 'SouthEastOutside');
% subplot(3,3,9);
% plot(rho_path',x_path(:,2:end));
% xlabel('\rho');
% ylabel('\beta(\rho)');
% xlim([min(rho_path),max(rho_path)]);
% print -depsc2 ../../manuscripts/notes/MandA_estimates.eps;
% print -dpng ../../manuscripts/notes/MandA_uncons.png;
% print -dpng ../../manuscripts/notes/MandA_intermediate.png;
% print -dpng ../../manuscripts/notes/MandA_constr.png;

%% plot the solution path
figure; hold on;
set(gca, 'FontSize', 20, 'XDir', 'reverse');
plot(rho_path',x_path(:,2:end));
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
% print -depsc2 ../../manuscripts/notes/MandA_solpath.eps;

%% Plot AIC and BIC

n = size(X,1);
p = size(x_path,2);
df_path = p-sum(abs(Aeq*x_path')<1e-6,1)-sum(abs(Aineq*x_path')<1e-6,1);
logL_path = zeros(1,size(x_path,1));
for i=1:size(x_path,1)
    logL_path(i) = glmfun(x_path(i,:)',X,y,wt,'binomial',[]);
end
AIC_path = logL_path + df_path;
BIC_path = logL_path + log(n)/2*df_path;

figure; hold on;
set(gca, 'FontSize', 20);
plot(rho_path, AIC_path);
plot(rho_path, BIC_path, '-.');
xlabel('\rho');
legend('AIC', 'BIC');
xlim([min(rho_path),max(rho_path)]);
%ylabel('BIC(\rho)');
% print -depsc2 ../../manuscripts/notes/MandA_AICBICpath.eps;