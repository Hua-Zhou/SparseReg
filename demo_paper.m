%% Examples in Zhou, Armagan and Dunson (2011)
% A demonstration of examples in the paper titled "Path Following and
% Empirical Bayes Model Selection for Sparse Regressions" by Zhou, Armagan,
% and Dunson (2011)

%% Prostate cancer data set - solution paths
% Load data set
clear;
printfig = false;           % print figure to eps file
fid = fopen('../../datasets/prostate.txt');
prostate = textscan(fid, ['%d',repmat('%f',1,9),'%s'], 'HeaderLines', 1, ...
    'CollectOutput', true);
fclose(fid);
names = {'lcavol', 'lweight', 'age', 'lbph', 'svi',	'lcp', 'gleason', ...
    'pgg45', 'lpsa'};
trainidx = strcmp(prostate{3},'T');     % index for training sample
prostate = prostate{2};     % 97-by-9 data matrix
X = prostate(trainidx,1:8); % training design matrix
X = bsxfun(@minus, X, mean(X,1));   % scale predictors to have mean 0
X = bsxfun(@rdivide, X, std(X));    % and variance 96\
X = [ones(size(X,1),1), X]; % add intercept
y = prostate(trainidx,end); % training response variable
% prepare the test data set
Xtest = prostate(~trainidx,1:8);
Xtest = bsxfun(@minus, Xtest, mean(Xtest,1));
Xtest = bsxfun(@rdivide, Xtest, std(Xtest));
Xtest = [ones(size(Xtest,1),1),Xtest];
ytest = prostate(~trainidx,end);

%% 
% Compute and display solution paths from different penalties
penalty = {'enet' 'enet' 'enet' 'power' 'power' 'log' 'log' 'mcp' 'scad'};
penparam = [1 1.5 2 0.5 1 0 1 0.1 3.7];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized

figure;
for i=1:length(penalty)
    tic;
    [rho_path,beta_path] = ...
        lsq_sparsepath(X,y,'penidx',penidx,'penalty',penalty{i}, ...
        'penparam',penparam(i));
    timing = toc;
    subplot(3,3,i);
    plot(rho_path,beta_path(2:end,:));
    if (i==8)
        xlabel('\rho');
    end
    if (i==4)
        ylabel('\beta(\rho)');
    end
    xlim([min(rho_path),max(rho_path)]);
    ylim([-0.3,0.85]);
    title([penalty{i} '(' num2str(penparam(i)) '), ' num2str(timing,1) 's']);
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/prostate_solpath', '.eps']);
end

%% Prostate cancer data set - power/enet penalty
% compute solution paths from power and enet penalties
penalty = {'power' 'power' 'power' 'power' 'enet' 'enet' 'enet' 'enet' 'enet'};
penparam = [0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
beta_all = cell(length(penalty),1);
rho_all = cell(length(penalty),1);
tic;
for i=1:length(penalty)
    [rho_path,beta_path] = lsq_sparsepath(X,y,'penidx',penidx, ...
        'penalty',penalty{i},'penparam',penparam(i));
    beta_all{i} = beta_path;
    rho_all{i} = rho_path;
end
timing = toc;
yhat_path = cellfun(@(B) X*B, beta_all, 'UniformOutput', false);
R2_path = cellfun(@(yhat) var(yhat)/var(yhat(:,end)), yhat_path, ...
    'UniformOutput', false);
modelsize_path = cellfun(@(B) sum(abs(B)>1e-6), beta_all, ...
    'UniformOutput', false);
%%
% plot $R^2$ vs model size
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(R2_path),1);
for i=1:length(R2_path)
	plot(modelsize_path{i}, R2_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 10]);
ylim([-0.05,1.05]);
xlabel('# predictors');
ylabel('adjusted R^2');
title(['power/enet, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'southeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/prostate_R2_power', '.eps']);
end
%%
% plot prediction MSE vs model size
mse_path = cellfun(@(B) var(repmat(ytest,1,size(B,2))-Xtest*B), beta_all, ...
    'UniformOutput', false);
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(mse_path),1);
for i=1:length(mse_path)
	plot(modelsize_path{i}, mse_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 10]);
ylim([0.4,1.1]);
xlabel('# predictors');
ylabel('prediction MSE');
title(['power/enet, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'northeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/prostate_mse_power', '.eps']);
end

%% Prostate cancer data set - log penalty
% compute solution paths from log penalties
penalty = {'log' 'log' 'log' 'log' 'log'};
penparam = [0.1 0.5 1 2 0];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
beta_all = cell(length(penalty),1);
rho_all = cell(length(penalty),1);
tic;
for i=1:length(penalty)
    [rho_path,beta_path] = ...
        lsq_sparsepath(X,y,'penidx',penidx,'penalty',penalty{i},...
        'penparam',penparam(i));
    beta_all{i} = beta_path;
    rho_all{i} = rho_path;
end
timing = toc;
yhat_path = cellfun(@(B) X*B, beta_all, 'UniformOutput', false);
R2_path = cellfun(@(yhat) var(yhat)/var(yhat(:,end)), yhat_path, ...
    'UniformOutput', false);
modelsize_path = cellfun(@(B) sum(abs(B)>1e-6), beta_all, ...
    'UniformOutput', false);
%%
% plot $R^2$ vs model size
figure; 
set(gca,'FontSize',15);
penparam_str = cell(length(R2_path),1);
for i=1:length(R2_path)
	plot(modelsize_path{i}, R2_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 10]);
ylim([-0.05,1.05]);
xlabel('# predictors');
ylabel('Adjusted R^2');
title(['log penalty, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'southeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/prostate_R2_log', '.eps']); %#ok<*UNRCH>
end
%%
% plot prediction MSE vs model size
mse_path = cellfun(@(B) var(repmat(ytest,1,size(B,2))-Xtest*B), beta_all, ...
    'UniformOutput', false);
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(mse_path),1);
for i=1:length(mse_path)
	plot(modelsize_path{i}, mse_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 10]);
ylim([0.4,1.1]);
xlabel('# predictors');
ylabel('prediction MSE');
title(['log penalty, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'northeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/prostate_mse_log', '.eps']);
end

%% South Africa heart disease data - solution paths
% read in data
clear;
printfig = false;
fid = fopen('../../datasets/saheart.txt');
rawdata = textscan(fid, [repmat('%f ', 1, 5) '%s ' repmat('%f ', 1, 6)],...
    'HeaderLines', 1, 'delimiter', '\t', 'CollectOutput', true);
fclose(fid);
rawdata = [rawdata{1,1} cellfun(@(s) strcmp(s,'Present'),rawdata{1,2}), ...
    rawdata{1,3}];
% sbp, tobacco, ldl, famhist, obesity, alcohol, age
trainidx = logical(rawdata(:,end)); % train/test split
X = rawdata(trainidx,[2 3 4 6 8 9 10]);
X = bsxfun(@minus, X, mean(X,1));   % centering
X = bsxfun(@rdivide, X, std(X));    % standardize
X = [ones(size(X,1),1), X];         % add intercept
y = rawdata(trainidx,end-1);
n = size(X,1); 
% prepare the test data
Xtest = rawdata(~trainidx,[2 3 4 6 8 9 10]);
Xtest = bsxfun(@minus, Xtest, mean(Xtest,1));   % centering
Xtest = bsxfun(@rdivide, Xtest, std(Xtest));    % standardize
Xtest = [ones(size(Xtest,1),1),Xtest];          % add intercept
ytest = rawdata(~trainidx,end-1);

%%
% Compare solution paths from different penalties
model = 'logistic';
penalty = {'enet' 'enet' 'enet' 'power' 'power' 'log' 'log' 'mcp' 'scad'};
penparam = [1 1.5 2 0.5 1 0 1 0.1 3.7];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized

figure;
for i=1:length(penalty)
    tic;
    [rho_path,beta_path] = ...
        glm_sparsepath(X,y,model,'penidx',penidx,'penalty',penalty{i},...
        'penparam',penparam(i));
    timing = toc;
    subplot(3,3,i);
    plot(rho_path,beta_path(2:end,:));
    if (i==8)
        xlabel('\rho');
    end
    if (i==4)
        ylabel('\beta(\rho)');
    end
    xlim([min(rho_path),max(rho_path)]);
    ylim([-0.3,0.85]);
    title([penalty{i} '(' num2str(penparam(i)) '), ' num2str(timing,1) 's']);
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/saheart_solpath', '.eps']);
end

%% South Africa heart disease data set - power/enet penalty
% compute solution paths from power and enet penalties
penalty = {'power' 'power' 'power' 'power' 'enet' 'enet' 'enet' 'enet' 'enet'};
penparam = [0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
beta_all = cell(length(penalty),1);
rho_all = cell(length(penalty),1);
tic;
for i=1:length(penalty)
    [rho_path,beta_path] = ...
        glm_sparsepath(X,y,model,'penidx',penidx,'penalty',penalty{i},...
        'penparam',penparam(i));
    beta_all{i} = beta_path;
    rho_all{i} = rho_path;
end
timing = toc;
yhat_path = cellfun(@(B) glmval(B,X(:,2:end),'logit'), beta_all, ...
    'UniformOutput', false);
dev_path = cellfun(@(yhat) 2*sum(log(bsxfun(@times,yhat,y) ...
    + bsxfun(@times,1-yhat,1-y))), yhat_path, 'UniformOutput', false);
modelsize_path = cellfun(@(B) sum(abs(B)>1e-6), beta_all, ...
    'UniformOutput', false);
%%
% plot deviance vs model size
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(dev_path),1);
for i=1:length(dev_path)
	plot(modelsize_path{i}, dev_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 8]);
xlabel('# predictors');
ylabel('- deviance');
title(['power/enet, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'southeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/saheart_dev_power', '.eps']);
end
%%
% plot prediction MSE vs model size
ypred_path = cellfun(@(B) glmval(B,Xtest(:,2:end),'logit'), beta_all, ...
    'UniformOutput', false);
mse_path = cellfun(@(ypred) sqrt(sum(bsxfun(@minus,ytest,ypred).^2)/length(ytest)), ...
    ypred_path, 'UniformOutput', false);
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(mse_path),1);
for i=1:length(mse_path)
	plot(modelsize_path{i}, mse_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 8]);
ylim([0.42 0.475]);
xlabel('# predictors');
ylabel('prediction MSE');
title(['power/enet, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'northeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/saheart_mse_power', '.eps']);
end
%% South Africa heart disease data set - log penalty
% compute solution paths from power and enet penalties
penalty = {'log' 'log' 'log' 'log' 'log'};
penparam = [0.1 0.5 1 2 0];
penidx = [false; true(size(X,2)-1,1)];  % leave intercept unpenalized
beta_all = cell(length(penalty),1);
rho_all = cell(length(penalty),1);
tic;
for i=1:length(penalty)
    [rho_path,beta_path] = ...
        glm_sparsepath(X,y,model,'penidx',penidx,'penalty',penalty{i},...
        'penparam',penparam(i));
    beta_all{i} = beta_path;
    rho_all{i} = rho_path;
end
timing = toc;
yhat_path = cellfun(@(B) glmval(B,X(:,2:end),'logit'), beta_all, ...
    'UniformOutput', false);
dev_path = cellfun(@(yhat) 2*sum(log(bsxfun(@times,yhat,y) ...
    + bsxfun(@times,1-yhat,1-y))), yhat_path, 'UniformOutput', false);
modelsize_path = cellfun(@(B) sum(abs(B)>1e-6), beta_all, ...
    'UniformOutput', false);
%%
% plot deviance vs model size
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(dev_path),1);
for i=1:length(dev_path)
	plot(modelsize_path{i}, dev_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 8]);
xlabel('# predictors');
ylabel('- deviance');
title(['log, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'southeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/saheart_dev_log', '.eps']);
end
%% 
% plot prediction MSE vs model size
ypred_path = cellfun(@(B) glmval(B,Xtest(:,2:end),'logit'), beta_all, ...
    'UniformOutput', false);
mse_path = cellfun(@(ypred) sqrt(sum(bsxfun(@minus,ytest,ypred).^2)/length(ytest)), ...
    ypred_path, 'UniformOutput', false);
figure;
set(gca,'FontSize',15);
set(gcf,'DefaultAxesLineStyleOrder','-|-.|--|:');
penparam_str = cell(length(mse_path),1);
for i=1:length(mse_path)
	plot(modelsize_path{i}, mse_path{i});
    penparam_str{i} = [penalty{i} '(' num2str(penparam(i)) ')'];
    hold all;
end
xlim([1 8]);
ylim([0.42 0.475]);
xlabel('# predictors');
ylabel('prediction MSE');
title(['log, ', num2str(timing,2) 's']);
legend(penparam_str, 'location', 'northeast');
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/saheart_mse_log', '.eps']);
end

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
penparam = 0.5;
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
    print('-depsc2', ['../../manuscripts/notes/manda_solpath_power', '.eps']);
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