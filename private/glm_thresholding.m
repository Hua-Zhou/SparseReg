function [xmin] = glm_thresholding(X,C,y,wt,lambda,pentype,penparam,model)
% GLM_THRESHOLDING Performs univariate GLM thresholding
%   argmin loss(x) + pen(abs(x),lambda)
%
% INPUT
%   X - n-by-p predictor vectors
%   C - n-by-1 constant vector
%   y - n-by-1 response vector
%   wt - n-by-1 weight vector
%   lambda - penalty constant (>=0)
%   penname - 'enet'|'log'|'mcp'|'power'|'scad'
%   penargs - index parameter for penalty function penname; allowed range
%       enet [1,2] (1 by default), log (0,inf) (1 by default), mcp (0,inf) 
%       (1 by default), power (0,2] (1 by default), scad (2,inf) (3.7 by default)
%   model - GLM model specifier
%
% OUTPUT
%   xmin(j) - argmin loss(X(:,j),y,wt) + pen(abs(x),lambda)
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu), Artin Armagan

% check proper input arguments
n = size(X,1);
if (numel(C)~=n)
    error('X and C have incompatible sizes');
elseif (size(C,1)==1)
    C = C';
end

if (length(y)~=n)
    error('y has incompatible size');
elseif (size(y,1)==1)
    y = y';
end

if (isempty(wt))
    wt = ones(n,1);
elseif (length(wt)~=n)
    error('wt has incompatible size');
elseif (any(wt<=0))
    erro('wt should be positive');
end

if (lambda<0)
    error('penalty constant lambda should be nonnegative');
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
    error('penalty type not recogonized. ENET|LOG|MCP|POWER|SCAD accepted');
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
    error('model not recogonized. LOGISTIC|POISSON accepted');
end

% call the mex function
xmin = glmthresholding(X,C,y,wt,lambda,penparam,pentype,model);

end