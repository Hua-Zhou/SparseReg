function [rho_path, beta_path] = glm_regpath(X,y,wt,D,pentype,penparam,model)
% LSQ_REGPATH Calculate the solution path of  
%   argmin loss(beta)+sum(penfun(D*beta))
%
% INPUT:
%   X - n-by-p design matrix
%   y - n-by-1 responses
%   wt - n-by-1 weights; wt = ones(p,1) if not supplied
%   D - m-by-p regularization matrix
%
% OUTPUT:
%   rho_path - rhos along the path
%   beta_path - solution vectors at each rho

% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou, hua_zhou@ncsu.edu

% check dimensions of inputs
[n,p] = size(X);

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

m = size(D,1);
if (size(D,2)~=p)
    error('regularization matrix D must be a m-by-p');
end
% Check rank of D
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
maxpreds = [];
penidx = [true(m,1); false(p-rankD,1)];
[rho_path,beta_path] = ...
    glm_sparsepath(X*T,y,wt,penidx,maxpreds,pentype,penparam,model);

% transform from new variables back to beta
beta_path = T*beta_path;

end