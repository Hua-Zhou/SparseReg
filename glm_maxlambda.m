function [maxlambda] = glm_maxlambda(x,c,y,wt,pentype,penparam,model)
%GLM_MAXLAMBDA Find the max lambda such that
%       argmin loss(beta*x+c) + pen(abs(beta),lambda)
%   is nonzero
%
% INPUT
%   x: n-by-1 predictor vector
%   c: n-by-1 constant vector
%   y: n-by-1 response vector
%   pentype - 'enet'|'log'|'mcp'|'power'|'scad'
%   penargs - index parameter for penalty function penname; allowed range
%       enet [1,2] (1 by default), log (0,inf) (1 by default), mcp (0,inf) 
%       (1 by default), power (0,2] (1 by default), scad (2,inf) (3.7 by default)
%   model - GLM model "logistic"|"loglinear"
%
% OUTPUT
%   maxlambda: max lambda such that argmin loss(x)+pen(abs(x),lambda)
%       becomes nonzero
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu), Artin Armagan
% RELEASE DATE: ??/??/????

% check proper input arguments
if (size(x,1)>1 && size(x,2)>1)
    error('x must be a vector');
else
    n = length(x);
end

if (size(c,1)>1 && size(c,2)>1)
    error('c must be a vector');
elseif (isempty(c))
    c = zeros(n,1);
elseif (length(c)~=n)
    error('size of c is incompatible with x');
end

if (size(y,1)>1 && size(y,2)>1)
    error('y must be a vector');
elseif (length(y)~=n)
    error('size of y is incompatible with x');
end

if (size(wt,1)>1 && size(wt,2)>1)
    error('wt must be a vector');
elseif (isempty(wt))
    wt = ones(n,1);
elseif (length(wt)~=n)
    error('size of wt is incompatible with x');
elseif (any(wt<=0))
    error('weights wt should be positive');
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