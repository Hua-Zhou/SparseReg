%% Simulation study for a large p small n problem
% 
clear;

n = 100;
p = 10000;
b = zeros(p+1,1);           % true signal
b(2:6) = 3;                 % first 5 predictors are 3
b(7:11) = -3;               % next 5 predictors are -3
truemodel = false(p,1);
truemodel(1:10) = true;
reps = 1;
penalty = 'power';
penparam = [0.25 0.5 0.75 1];
penidx = [false; true(p,1)];
maxpreds = 76;

mse = zeros(reps,length(penparam),3);
runtime = zeros(reps,length(penparam),3);
fpr = zeros(reps,length(penparam),3);
fnr = zeros(reps,length(penparam),3);
rng(52570);               % seed
warning off all;
for replicate=1:reps
    
    disp('');
    display(['replicate ' num2str(replicate)]);
    X = randn(n,p);             % generate a random design matrix
    X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))); % normalize predictors
    X = [ones(size(X,1),1) X];  %#ok<AGROW> % add intercept
    y1 = X*b+randn(n,1);        % response for linear model
    inner = X*b;                % linear part
    y2 = poissrnd(exp(inner));  % response for Poisson model
    prob = 1./(1+exp(-inner));
    y3 = binornd(1,prob);       % response for logistic model
    
    for i=1:length(penparam)

        display([penalty ' ' num2str(penparam(i))]);
        % linear model
        tic;
        [~,beta_path,eb_path] = lsq_sparsepath(X,y1,'penidx',penidx, ...
            'maxpreds',maxpreds,'penalty',penalty,'penparam',penparam(i));
        timing = toc;
        display(['Linear reg: ' num2str(timing)]);        
        [~,ebidx] = min(eb_path);
        ebestimate = beta_path(:,ebidx);
        ebmodel = abs(ebestimate(2:end))>1e-8;
        runtime(replicate,i,1) = timing;
        mse(replicate,i,1) = norm(ebestimate-b)/sqrt(size(beta_path,1));
        TP = nnz(ebmodel & truemodel);
        FP = nnz(ebmodel & ~truemodel);
        TN = nnz(~ebmodel & ~truemodel);
        FN = nnz(~ebmodel & truemodel);
        fpr(replicate,i,1) = FP/(FP+TN);
        fnr(replicate,i,1) = FN/(TP+FN);
        
        % Poisson model
        tic;
        [~,beta_path,eb_path] = ...  % compute solution path
            glm_sparsepath(X,y2,'loglinear','penidx',penidx,'maxpreds',maxpreds, ...
            'penalty',penalty,'penparam',penparam(i));
        timing = toc;
        display(['Poisson reg: ' num2str(timing)]);
        [~,ebidx] = min(eb_path);
        ebestimate = beta_path(:,ebidx);
        ebmodel = abs(ebestimate(2:end))>1e-8;
        runtime(replicate,i,2) = timing;
        mse(replicate,i,2) = norm(ebestimate-b)/sqrt(size(beta_path,1));
        TP = nnz(ebmodel & truemodel);
        FP = nnz(ebmodel & ~truemodel);
        FN = nnz(~ebmodel & truemodel);
        TN = nnz(~ebmodel & ~truemodel);
        fpr(replicate,i,2) = FP/(FP+TN);
        fnr(replicate,i,2) = FN/(TP+FN);
        
        % logistic model
        tic;
        [~,beta_path,eb_path] = ...  % compute solution path
            glm_sparsepath(X,y3,'logistic','penidx',penidx,'maxpreds',maxpreds, ...
            'penalty',penalty,'penparam',penparam(i));
        timing = toc;
        display(['Logistic reg: ' num2str(timing)]);
        [~,ebidx] = min(eb_path);
        ebestimate = beta_path(:,ebidx);
        ebmodel = abs(ebestimate(2:end))>1e-8;
        runtime(replicate,i,3) = timing;
        mse(replicate,i,3) = norm(ebestimate-b)/sqrt(size(beta_path,1));
        TP = nnz(ebmodel & truemodel);
        FP = nnz(ebmodel & ~truemodel);
        FN = nnz(~ebmodel & truemodel);
        TN = nnz(~ebmodel & ~truemodel);
        fpr(replicate,i,3) = FP/(FP+TN);
        fnr(replicate,i,3) = FN/(TP+FN);        
    end
end%for
warning on all;
timestamp = regexprep(datestr(now), ' |:', '-');
save(['sim-results-' timestamp '.mat']);

%% 
% post-processing

% load('sim-results-02-Jan-2012-17-20-27.mat');

printfig = false;
ylabels = {'Linear', 'Poisson', 'Logistic'};
% plot run times
figure; 
for i=1:size(runtime,3)
    subplot(size(runtime,3),1,i)
    if (i<size(runtime,3))
        boxplot(runtime(:,:,i));
        set(gca,'XTickLabel',{' '})
    else
        boxplot(runtime(:,:,i),'labels',penparam);
    end
    ylabel(ylabels{i});
    if (i==1)
        title('Run Time in Seconds');
    elseif (i==size(runtime,3))
        xlabel('Exponent of power penalty \eta');
    end
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/n100-p10000-timing', '.eps']);
end

%% plot false positive rate (FPR)
figure;
for i=1:size(fpr,3)
    subplot(size(fpr,3),1,i)
    if (i<size(fpr,3))
        boxplot(fpr(:,:,i));
        set(gca,'XTickLabel',{' '})
    else
        boxplot(fpr(:,:,i),'labels',penparam,'datalim',[0,0.006]);
    end
    ylabel(ylabels{i});
    if (i==1)
        title('FPR of Empirical Bayes Model');
    elseif (i==size(fpr,3))
        xlabel('Exponent of power penalty \eta');
    end
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/n100-p10000-fpr', '.eps']);
end

%% plot false positive rate (FNR)
figure;
for i=1:size(fnr,3)
    subplot(size(fnr,3),1,i)
    if (i<size(fnr,3))
        boxplot(fnr(:,:,i));
        set(gca,'XTickLabel',{' '})
    else
        boxplot(fnr(:,:,i),'labels',penparam);
    end
    ylabel(ylabels{i});
    if (i==1)
        title('FNR of Empirical Bayes Model');
    elseif (i==size(fnr,3))
        xlabel('Exponent of power penalty \eta');
    end
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/n100-p10000-fnr', '.eps']);
end

%% plot MSE
figure;
for i=1:size(mse,3)
    subplot(size(mse,3),1,i)
    if (i<size(mse,3))
        boxplot(mse(:,:,i));
        set(gca,'XTickLabel',{' '})
    else
        boxplot(mse(:,:,i),'labels',penparam,'datalim',[0,1]);
    end
    ylabel(ylabels{i});
    if (i==1)
        title('MSE of Emprical Bayes Model');
    elseif (i==size(mse,3))
        xlabel('Exponent of power penalty \eta');
    end
end
if (printfig)
    print('-depsc2', ['../../manuscripts/notes/n100-p10000-mse', '.eps']);
end
