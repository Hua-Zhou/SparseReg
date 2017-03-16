function [betahat] = lsq_sparsereg(X,y,lambda,varargin)
% LSQ_SPARSEREG Sparse linear regression at a fixed penalty value
%   BETAHAT = LSQ_SPARSEREG(X,y,lambda) fits penalized linear regression
%   using the predictor matrix X, response Y, and tuning parameter value
%   LAMBDA. The result BETAHAT is a vector of coefficient estimates. By
%   default it fits the lasso regression.
%
%   BETAHAT = LSQ_SPARSEREG(X,y,lambda,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit. Parameters are:
%
%       'maxiter' - maxmum number of iterations
%
%       'penidx' - a logical vector indicating penalized coefficients
%
%       'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%
%       'pentype' - ENET|LOG|MCP|POWER|SCAD
%
%       'sum_x_squares' - precomputed sum(wt*X.^2,1)
%
%       'weights' - a vector of prior weights
%
%       'x0' - a vector of starting point
%
%   See also LSQ_SPARSEPATH,GLM_SPARSEREG,GLM_SPARSEPATH.
%
%   Eexample
%
%   References
%
% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
[n,p] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('lambda', @(x) x>=0);
argin.addParamValue('maxiter', 1000, @(x) isnumeric(x) && x>0);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', [], @isnumeric);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);
argin.addParamValue('sum_x_squares', [], @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==p);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);
argin.addParamValue('x0', zeros(p,1), @(x) isnumeric(x) && length(x)==p);
% parse inputs
y = reshape(y,n,1);
argin.parse(X,y,lambda,varargin{:});
maxiter = round(argin.Results.maxiter);
penidx = reshape(argin.Results.penidx,p,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
sum_x_squares = argin.Results.sum_x_squares;
wt = reshape(argin.Results.weights,n,1);
x0 = reshape(full(argin.Results.x0),p,1);
% compute covariate norms if not supplied
if (isempty(sum_x_squares))
    sum_x_squares = sum(bsxfun(@times, wt, X.*X),1)';
else
    sum_x_squares = reshape(sum_x_squares,p,1);
end
% set up penalty parameter for penalty families
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
        penparam = 1;   % 1 by default
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
        penparam = 3.7; % 3.7 by default
    elseif (penparam<=2)
        error('index parameter for SCAD penalty should be larger than 2');
    end
else
    error('penalty type not recogonized. ENET|LOG|MCP|POWER|SCAD accepted');
end

% call the mex function
betahat = ...
    lsqsparse(x0,X,y,wt,lambda,sum_x_squares,penidx,maxiter,pentype,penparam);

end