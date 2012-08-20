%% M&A data - logistic regression with cubic trend filtering
% load data
clear;
printfig = false;
load '../../datasets/MandAcleanData.mat';
X = x; clear x;
display(names);
[~,dev_orig] = glmfit(X,y,'binomial');
% discretize the covariates into nbins bins; the first bin is used as the
% reference level; effect coding is used for all category variables
XDiscretized = cell(1,7);
nbins = 10;
for j=1:7
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
%% 
% cubic trend filtering on all predictors 1-7
% construct design matrix
X = ones(size(X,1),1);
for j=1:7
    X = [X XDiscretized{j}];
end
% construct penalty matrix for cubic trend filtering
nj = cellfun(@(A) size(A,2), XDiscretized);
D = zeros(0,size(X,2));
for j=1:7
    fusemat = -eye(nj(j));
    fusemat(nj(j)+1:(nj(j)+1):nj(j)^2-1) = 1;
    fusemat = fusemat^4;
    fusemat([end-3 end-2 end-1 end],:) = [];
    fusemat(end,:) = 0;
    fusemat(end,[end-2 end-1 end]) = [-1 2 -1];
    fusemat(end+1,:) = [3 0 ones(1,size(fusemat,2)-2)]; %#ok<*SAGROW>
    fusemat = [zeros(size(fusemat,1),sum(nj(1:(j-1)))+1) fusemat ...
        zeros(size(fusemat,1),sum(nj(j+1:end)))]; %#ok<*AGROW>
    D = [D; fusemat];
end
%%
% path algorithm
model = 'logistic';
penalty = 'power';          % set penalty function to log
penparam = 1;
tic;
[rho_path,beta_path,eb_path] = glm_regpath(X,y,D,model,'penalty','log', ...
    'penparam',penparam);
timing = toc;
[~,ebidx] = min(eb_path);
%%
% plot solution path
figure;
plot(rho_path,beta_path);
xlabel('\rho');
ylabel('\beta(\rho)');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) '), ' num2str(timing,2) ' sec']);
line([rho_path(ebidx),rho_path(ebidx)],ylim);
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/manda_solpath_power', '.eps']); %#ok<*UNRCH>
end
%%
% plot empirical Bayes criterion along path
figure;
plot(rho_path(~isnan(eb_path)),eb_path(~isnan(eb_path)));
ylabel('EBC');
xlim([min(rho_path),max(rho_path)]);
title([penalty '(' num2str(penparam) ')']);
line([rho_path(ebidx),rho_path(ebidx)],ylim);
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/manda_ebcpath_power', '.eps']);
end
%%
% test the original model vs the fully regularized model
inner = X*beta_path(:,end);
prob = exp(inner)./(1+exp(inner));
dev_reg = -2*sum(log(y.*prob+(1-y).*(1-prob)));
pvalue = 1 - chi2cdf(dev_orig-dev_reg,22);
display(pvalue);
%% 
% plot unconstrained/constrained estimates
rhokink = 1;            % fully regularized estimate
[~,rhokink2] = min(eb_path);  % estiamte located by EBC
load '../../datasets/MandAcleanData.mat' x;
figure;
set(gca,'FontSize', 15);
for j=1:7
    subplot(3,3,j); hold on;
    xqt = [quantile(x(:,j),nbins-1) max(x(:,j))]';
    xidx = (2+sum(nj(1:j-1))):(1+sum(nj(1:j)));
    plot(xqt,[-sum(beta_path(xidx,end)) beta_path(xidx,end)'],'o');
    plot(xqt,[-sum(beta_path(xidx,rhokink2)) beta_path(xidx,rhokink2)'],':');
    plot(xqt,[-sum(beta_path(xidx,rhokink)) beta_path(xidx,rhokink)'],'-+');
    xlabel(names{j});
    xlim([min(x(:,j)) max(x(:,j))+abs(xqt(1))]);
end
legend('unconstrained estimate', ['\rho=' num2str(rho_path(rhokink2))], ...
    'constrained estimate', 'Location', 'SouthEastOutside');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/manda_estimates', '.eps']);
end
%%
% test the model located by EBC vs the fully regularized model
[~,bestebcidx] = min(eb_path);
inner = X*beta_path(:,bestebcidx);
prob = exp(inner)./(1+exp(inner));
dev_reg = -2*sum(log(y.*prob+(1-y).*(1-prob)));
pvalue = 1 - chi2cdf(dev_orig-dev_reg,22);
display(pvalue);