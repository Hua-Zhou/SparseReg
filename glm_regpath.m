function [rho_path, beta_path, eb_path] = glm_regpath(X,y,D,model,varargin)
% GLM_REGPATH Solution path of regularized GLM regression
%
%   [RHO_PATH,BETA_PATH,EB_PATH] = GLM_REGPATH(X,Y,D,MODEL) computes the
%   solution path of regularized GLM regression using the predictor matrix
%   X and response Y and regularization matrix D:
%   loss(beta)+penalty(D*beta,lambda). The result RHO_PATH holds rhos along
%   the patha. BETA_PATH holds solution vectors at each rho. EB_PATH holds
%   the empicial Bayes criteria at each rho. By default it fits the lasso
%   regularization.
%
%   [RHO_PATH,BETA_PATH] = GLM_REGPATH(X,Y,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit. Parameters are:
%
%       'penalty' - ENET|LOG|MCP|POWER|SCAD
%
%       'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%
%       'weights' - a vector of prior weights.
%
%   See also LSQ_REGPATH,LSQ_SPARSEPATH,GLM_SPARSEPATH.
%
%   References:
%

% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
[n,p] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(x)==n);
argin.addRequired('D', @(x) isnumeric(x) && size(x,2)==p);
argin.addRequired('model', @ischar);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', 1, @isnumeric);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);

% parse inputs
argin.parse(X,y,D,model,varargin{:});
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
wt = reshape(argin.Results.weights,n,1);
y = reshape(y,n,1);

% Check rank of D
m = size(D,1);
D = sparse(D);
[~,R,perm] = qr(D,0);
rankD = sum(abs(diag(R)) > abs(R(1))*max(m,p)*eps(class(R)));
if (m>p || rankD<m)
    error('regularization matrix D must have full row rank');
end

pentype = upper(pentype);
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
    error('penaty type not recogonized. ENET|LOG|MCP|POWER|SCAD accepted');
end

model = upper(model);
if (strcmp(model,'LOGISTIC'))
    if (any(y<0) || any(y>1))
        error('responses outside [0,1]');
    end
elseif (strcmp(model,'LOGLINEAR'))
    if (any(y<0))
        error('responses y must be nonnegative');
    end
else
    error('model not recogonized. LOGISTIC|LOGLINEAR accepted');
end

% V is the transformation matrix from beta to new variables
if rankD < p    % D is column rank deficient
    V = sparse(p,p);
    V(1:m,:) = D;
    V(m+1:end,perm(m+1:end)) = eye(p-rankD);
else
    V = D;
end
% T is the transformation matrix from new variables back to beta
T = (V'*V)\V';

% performa path following in new variables
penidx = [true(m,1); false(p-rankD,1)];
[rho_path,beta_path,eb_path] = glm_sparsepath(X*T,y,model,'weights',wt, ...
    'penidx',penidx,'penalty',pentype,'penparam',penparam);

% transform from new variables back to beta
beta_path = T*beta_path;

end