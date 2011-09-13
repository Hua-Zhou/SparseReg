function [betahat] = ...
    glm_sparsereg(X,y,wt,lambda,x0,penidx,maxiter,pentype,penparam,model)
% GLM_SPARSEREG Sparse GLM regression at a fixed penalty value
%   Compute argmin loss(beta) + penalty(beta(penidx),lambda)
%    
% INPUT
%   X - n-by-p design matrix
%   y - n-by-1 response vector
%   wt - n-by-1 weights; set to ones if empty
%   lambda - penalty constant (>=0)
%   x0 - p-by-1 initial estimate; set to zeros if empty
%   penidx - p-by-1 logical vector indicating penalized coefficients; 
%       set to trues if empty
%   maxiter - maxmum number of iterations; set to 1000 if empty
%   pentype - ENET|LOG|MCP|POWER|SCAD
%   penparam - index parameter for penalty; if empty, set to default values:
%       ENET, 1, LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%   model - GLM model specifiler: logistic|poisson
%
% OUTPUT
%   betahat - regression coefficient estimate
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou, hua_zhou@ncsu.edu
% RELEASE DATE: ??/??/????

% check proper input arguments
[n,p] = size(X);

if (isempty(x0))
    x0 = zeros(p,1);
elseif (numel(x0)~=p)
    error('x0 has incompatible size');
elseif (size(x0,1)==1)
    x0 = x0';
end

if (numel(y)~=n)
    error('y has incompatible size');
elseif (size(y,1)==1)
    y = y';    
end

if (isempty(wt))
    wt = ones(n,1);
elseif (numel(wt)~=n)
    error('wt has incompatible size');
elseif (size(wt,1)==1)
    wt = wt';
elseif (any(wt<=0))
    error('weights wt should be positive');    
end

if (lambda<0)
    error('penalty constant lambda should be nonnegative');
end

if (isempty(penidx))
    penidx = true(p,1);
elseif (numel(penidx)~=p)
    error('penidx has incompatible size');
elseif (size(penidx,1)==1)
    penidx = penidx';
end

if (isempty(maxiter))
    maxiter = 1000;
elseif (maxiter<=0)
    error('maxiter should be a positive integer');
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
betahat = ...
    glmsparse(x0,X,y,wt,lambda,penidx,maxiter,pentype,penparam,model);

end