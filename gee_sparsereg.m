function [betahat,alphahat,stats] ...
    = gee_sparsereg(id,time,X,y,model,workcorr,lambda,varargin)
% GEE_SPARSEREG Sparse GEE regression at a fixed penalty value
%   [BETAHAT,ALPHAHAT,STATS] =
%   GEE_SPARSEREG(ID,TIME,X,Y,MODEL,WORKCORR,LAMBDA) fits penalized GEE
%   regression using the predictor matrix X, response Y, working
%   correlation structure WORKCORR, and tuning parameter value LAMBDA.
%   MODEL specifies the model: 'normal', 'logistic' or 'loglinear'. The
%   result BETAHAT is a vector of coefficient estimates, and ALPHAHAT is
%   the estimates for working correlation. By default it fits the lasso
%   regression.
%
% INPUT:
%   'id' - id of subject
%   'time' - time pts of multiple measurements of same id
%   'X' - n-by-p design matrix
%   'y' - n-by-1 responses
%   'model'- 'normal','logistic',or'loglinear'
%   'workcorr' - 'equicorr', 'AR1', 'Markov', 'tridiag','unstructured',or
%       'indep'
%   'lambda' - regularization tuning parameter
%
% OPTIONAL NAME-VALUE PAIRS:
%   'geeMaxIter' - maxmum number of GEE iterations
%   'maxiter' - maxmum number of penalized least square iterations
%   'penidx' - a logical vector indicating penalized coefficients
%   'penalty' - ENET|LOG|MCP|POWER|SCAD
%   'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%   'tolX' - tolerance of relative change in betahat parameter values
%   'weights' - a vector of prior weights
%   'b0' - a vector of starting point
%
% OUTPUT:
%   'betahat' - estimated regression coefficients
%   'alphahat' - estimated working correlation parameters
%   'stats' - algorithmic statistics
%
% See also LSQ_SPARSEPATH,LSQ_SPARSEREG,GLM_SPARSEPATH.
%
% References:
%

% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
[n,p] = size(X);
argin = inputParser;
argin.addRequired('id', @isnumeric);
argin.addRequired('time', @isnumeric);
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('model', @(x) strcmpi(x,'normal') || ...
    strcmpi(x,'logistic')||strcmpi(x,'loglinear'));
argin.addRequired('workcorr', @(x) strcmpi(x,'equicorr') || strcmpi(x,'AR1') ...
    || strcmpi(x,'Markov') || strcmpi(x,'tridiag') ...
    || strcmpi(x,'unstructured') || strcmpi(x,'indep'));
argin.addRequired('lambda', @(x) x>=0);
argin.addParamValue('geeMaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('maxiter', 1000, @(x) isnumeric(x) && x>0);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', [], @isnumeric);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);
argin.addParamValue('tolX', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);
argin.addParamValue('b0', zeros(p,1), @(x) isnumeric(x) && length(x)==p);

% parse inputs
y = reshape(y,n,1);
argin.parse(id,time,X,y,model,workcorr,lambda,varargin{:});
nGEEMaxIter = round(argin.Results.geeMaxIter);
maxiter = round(argin.Results.maxiter);
penidx = reshape(argin.Results.penidx,p,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
tolX = argin.Results.tolX;
wt = reshape(argin.Results.weights,n,1);
b0 = reshape(full(argin.Results.b0),p,1);
if (strcmp(pentype,'ENET'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<1 || penparam>2)
        error('index parameter for ENET penalty should be in [1,2]');
    end
elseif (strcmp(pentype,'LOG'))
    if (isempty(penparam))
        penparam = 1;
    elseif (penparam<0)
        error('index parameter for LOG penalty should be nonnegative');
    end
elseif (strcmp(pentype,'MCP'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<=0)
        error('index parameter for MCP penalty should be positive');
    end
elseif (strcmp(pentype,'POWER'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<=0 || penparam>2)
        error('index parameter for POWER penalty should be in (0,2]');
    end
elseif (strcmp(pentype,'SCAD'))
    if (isempty(penparam))
        penparam = 3.7;
    elseif (penparam<=2)
        error('index parameter for SCAD penalty should be larger than 2');
    end
else
    error('penalty type not recogonized. ENET|LOG|MCP|POWER|SCAD accepted');
end

% check model
model = upper(model);
if strcmp(model,'NORMAL')
elseif strcmp(model,'LOGISTIC')
    if (any(y<0) || any(y>1))
        error('responses outside [0,1]');
    end
elseif strcmp(model,'LOGLINEAR')
    if (any(y<0))
        error('responses y must be nonnegative');
    end
else
    error('model not recogonized. LOGISTIC|POISSON accepted');
end

% compute covariate norms if not supplied
sum_x_squares = sum(bsxfun(@times, wt, X.*X),1)';

% sort data according to ID/TIME
[dummy, sortIdx]= sortrows([id time]); %#ok<ASGLU>
idSort = id(sortIdx);
timeSort = time(sortIdx);
nTimePts = length(unique(time));
xSort = X(sortIdx,:);
ySort = y(sortIdx);
cluster = unique(idSort); % cluster is sorted from unique()
nCluster = length(cluster);
clusterSize = histc(idSort, cluster);
% cluster index matrix
clusterIndex = false(n, nCluster);
for iCluster = 1:nCluster
    clusterIndex(:, iCluster) = (idSort == iCluster);
end

% start point by ignoring correlation structure
if lambda==0
    betahat = bsxfun(@times, xSort, wt) \ (ySort.*wt);
else
    betahat = ...
        lsqsparse(b0,xSort,ySort,wt,lambda,sum_x_squares,...
        penidx,maxiter,pentype,penparam);
end

% main GEE loop
xWork = zeros(size(X));
yWork = zeros(size(y));
for iGEEIter = 1:nGEEMaxIter
    
    % update Pearson residuals and organize in cell array
    if strcmp(model, 'NORMAL')
        mu = xSort * betahat;   % E(y) = mu
        resid = y - mu;
        asqrt = ones(n,1);      % Var(y) = phi*diag(a)
    elseif strcmp(model, 'LOGISTIC')
        eta = xSort * betahat;   
        mu = 1 ./ (exp(-eta) + 1);  % E(y) = mu
        mu(eta>30) = 1;
        mu(eta<-30) = 0;
        resid = y - mu;
        asqrt = sqrt(mu.*(1-mu)); % Var(y) = phi*diag(a)
    elseif strcmp(model, 'LOGLINEAR')
        mu = exp(xSort*betahat);  % E(y) = mu
        resid = y - mu;
        asqrt = sqrt(mu); % Var(y) = phi*diag(a)
    end
    residCell = mat2cell(resid, clusterSize, 1);
    
    % estimate correlation parameters and transform data
    pe = nnz(betahat);  % effective model size
    if pe >= n; pe = 0; end;
    phihat = norm(resid)^2 / (n-pe);
    if strcmpi(workcorr, 'equicorr')
        
        % estimate alpha
        alphahat = sum( cellfun(@(x) sum(x)^2 - norm(x)^2, residCell, ...
            'UniformOutput', true) );
        alphahat = alphahat/phihat/(sum(clusterSize.*(clusterSize-1))-pe);
        % TODO: make sure alphahat is in (0,1)
        
        % transform data
        for iCluster = 1:nCluster
            Vi = repmat(alphahat, ...
                clusterSize(iCluster), clusterSize(iCluster));
            Vi(1:clusterSize(iCluster)+1:clusterSize(iCluster)^2) = 1;
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster))');
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster)));
            Vi = chol(Vi);
            yWork(clusterIndex(:,iCluster)) ...
                = Vi \ ySort(clusterIndex(:,iCluster));
            xWork(clusterIndex(:,iCluster),:) ...
                = Vi \ xSort(clusterIndex(:,iCluster), :);
        end
                
    elseif strcmpi(workcorr, 'AR1')
        
        % estimate alpha
        alphahat = sum( cellfun(@(x) sum(x(2:end).*x(1:end-1)), residCell, ...
            'UniformOutput', true) );
        alphahat = alphahat / phihat / (sum(clusterSize-1)-pe);
        % TODO: make sure alpha is in (0,1)
        
        % transform data
        for iCluster = 1:nCluster
            Vi = alphahat .^ abs( bsxfun(@minus, (1:clusterSize(iCluster))', ...
                1:clusterSize(iCluster)) ); 
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster))');
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster)));
            Vi = chol(Vi);
            yWork(clusterIndex(:,iCluster)) ...
                = Vi \ ySort(clusterIndex(:,iCluster));
            xWork(clusterIndex(:,iCluster),:) ...
                = Vi \ xSort(clusterIndex(:,iCluster), :);
        end
        
    elseif strcmpi(workcorr, 'Markov')
        
    elseif strcmpi(workcorr, 'tridiag')
        
        % estimate alpha
        alphahat = sum( cellfun(@(x) sum(x(2:end).*x(1:end-1)), residCell, ...
            'UniformOutput', true) );
        alphahat = alphahat / phihat / (sum(clusterSize-1)-pe);
        % TODO: make sure alpha is in (0,1)

        % transform data
        for iCluster = 1:nCluster
            Vi = eye(clusterSize(iCluster));
            Vi(2:size(Vi,1)+1:(numel(Vi)-(size(Vi,1)))) = alphahat;
            Vi(size(Vi,1)+1:size(Vi,1)+1:(numel(Vi)-1)) = alphahat;
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster))');
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster)));
            try 
                Vi = chol(Vi);
            catch err
                % project to the nearest correlation matrix
                if strcmp(err.identifier,'MATLAB:posdef')
                    Vi = nearcorr(Vi,[],[],[],[],[],[]);
                    Vi = Vi + 1e-8*eye(size(Vi));
                    %Vi = nearcorr(Vi,[],[],[],[],[],[]);
                    Vi = chol(Vi);
%                     [evecVi, evalVi] = eig(Vi);
%                     evalVi = diag(evalVi);
%                     tolVi = eps(evalVi(1))*max(size(Vi));
%                     Vi = bsxfun(@times, evecVi(:,evalVi>tolVi)', ...
%                         sqrt(evalVi(evalVi>tolVi)));
                else
                    rethrow(err);
                end
            end
                  
            yWork(clusterIndex(:,iCluster)) ...
                = Vi \ ySort(clusterIndex(:,iCluster));
            xWork(clusterIndex(:,iCluster),:) ...
                = Vi \ xSort(clusterIndex(:,iCluster), :);

        end

    elseif strcmpi(workcorr, 'unstructured')
        
        % estimate alpha
        alphahat = zeros(nTimePts, nTimePts);
        for iCluster = 1:nCluster
            alphahat(timeSort(clusterIndex(:,iCluster)), ...
                timeSort(clusterIndex(:,iCluster))) ...
                = alphahat(timeSort(clusterIndex(:,iCluster)), ...
                timeSort(clusterIndex(:,iCluster))) ...
                + residCell{iCluster}*residCell{iCluster}';
        end
        if pe >= nCluster; pe = 0; end;
        alphahat = alphahat / phihat / (nCluster-pe);
        % TODO: make sure alpha is in (0,1)
        
        % transform data
        for iCluster = 1:nCluster
            Vi = alphahat(timeSort(clusterIndex(:,iCluster)), ...
                timeSort(clusterIndex(:,iCluster)));
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster))');
            Vi = bsxfun(@times, Vi, asqrt(clusterIndex(:,iCluster)));
            Vi = chol(Vi);
            yWork(clusterIndex(:,iCluster)) ...
                = Vi \ ySort(clusterIndex(:,iCluster));
            xWork(clusterIndex(:,iCluster),:) ...
                = Vi \ xSort(clusterIndex(:,iCluster), :);
        end
        
    elseif strcmpi(workcorr, 'indep')
        
        % no extra work for correlation structure
        alphahat = [];
        yWork = ySort ./ asqrt;
        xWork = bsxfun(@times, xSort, 1./asqrt);
        
    end
    
    % update mean parameters
    betaOld = betahat;
    if lambda==0
        betahat = bsxfun(@times, xWork, wts) \ (yWork.*wts);
    else
        betahat = ...
            lsqsparse(betaOld,xWork,yWork,wt,lambda,sum(xWork.^2,1),...
            penidx,maxiter,pentype,penparam);
    end
    
    % stopping criteria
    if norm(betahat-betaOld) < tolX * (norm(betaOld)+1)
        break;
    end

end

% collect some algorithmic statistics
stats.iterations = iGEEIter;


end
