function [maxlambda] = glm_maxlambda(x,y,model,varargin)
% GLM_MAXLAMBDA Max lambda for the solution path of sparse GLM
%
%   MAXLAMBDA = GLM_MAXLAMBDA(X,Y,MODEL) returns the
%   max lambda of solution path for penalized GLM. X is the covariate
%   vector. Y is the response vector. MODEL indicates the GLM model:
%   'logistic' or 'loglinear'. The result MAXLAMBDA is the max lambda such
%   that argmin loss(b)+pen(abs(b),lambda) is nonzero. By default, it
%   assumes lasso solution path.
%
%   MAXLAMBDA = GLM_MAXLAMBDA(X,Y,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit. Parameters are:
%
%       'offset' - a vector of offset constants.
%
%       'penalty' - ENET|LOG|MCP|POWER|SCAD
%
%       'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%
%       'weights' - a vector of prior weights.
%
%   See also LSQ_MAXLAMBDA.
%
%   References:
%

% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
if (size(x,1)>1 && size(x,2)>1)
    error('x must be a vector');
else
    n = length(x);
end
argin = inputParser;
argin.addRequired('x', @isnumeric);
argin.addRequired('y', @(x) length(x)==n);
argin.addRequired('model', @(x) strcmpi(x,'logistic')||strcmpi(x,'loglinear'));
argin.addParamValue('offset', zeros(n,1), @(x) length(x)==n);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', [], @isnumeric);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) ...
    && length(x)==n);

% parse inputs
x = reshape(x,n,1);
y = reshape(y,n,1);
argin.parse(x,y,model,varargin{:});
c = reshape(argin.Results.offset,n,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
wt = reshape(argin.Results.weights,n,1);

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

model = upper(model);
if (strcmp(model,'LOGISTIC'))
    if (any(y<0) || any(y>1))
        error('responses y outside [0,1]');
    end
elseif (strcmp(model,'LOGLINEAR'))
    if (any(y<0))
        error('responses y must be nonnegative');
    end
else
    error('model not recogonized. LOGISTIC|LOGLINEAR accepted');
end

% call the mex function
maxlambda = glmmaxlambda(x,c,y,wt,pentype,penparam,model);

end