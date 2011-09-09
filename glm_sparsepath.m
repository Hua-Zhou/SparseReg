function [rho_path,beta_path,rho_kinks,fval_kinks] = ...
    glm_sparsepath(X,y,wt,penidx,maxpreds,pentype,penparam,model)
% GLM_SPARSEREG Solution path for sparse GLM regression
%   argmin loss(beta) + rho*sum(penfun(beta(penidx)))
%
% INPUT:
%   X - n-by-p design matrix
%   y - n-by-1 responses
%   wt - n-by-1 weights; wt = ones(p,1) if not supplied
%   penidx - p-by-1 logical index of the coefficients being penalized;
%       penidx = true(p,1) if not supplied
%   maxpreds - maximum number of top predictors requested; maxpreds=min(n,p)
%       if not supplied
%   penname - 'enet'|'log'|'mcp'|'power'|'scad'
%   penargs - index parameter for penalty function penname; allowed range
%       enet [1,2] (1 by default), log (0,inf) (1 by default), mcp (0,inf) 
%       (1 by default), power (0,2] (1 by default), scad (2,inf) (3.7 by default)
%
% Output:
%   rho_path - rhos along the path
%   x_path - solution vectors at each rho
%   rho_kinks - kinks of the solution paths
%   fval_kinks - objective values at kinks
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu), Artin Armagan
% RELEASE DATE: ??/??/????

% check arguments
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

if (isempty(penidx))
    penidx = true(p,1);
elseif (numel(penidx)~=p)
    error('penidx has incompatible size');
elseif (size(penidx,1)==1)
    penidx = penidx';
end

if (isempty(maxpreds) || maxpreds>=min(n,p))
    maxpreds = min(n,p);
elseif (maxpreds<=0)
    error('maxpreds should be a positive integer');
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
else
    error('model not recogonized. LOGISTIC|POISSON accepted');
end

% precompute and allocate storage for path
tiny = 1e-4;
islargep = p>=1000;
if (islargep)
    beta_path = sparse(p,1);
else
    beta_path = zeros(p,1);
end
rho_path = 0;

% set up ODE solver and unconstrained optimizer
maxiters = 2*min([n,p]);    % max iterations for path algorithm
maxrounds = 3;              % max iterations for lsq_sparsereg
refine = 1;
odeopt = odeset('Events',@events, 'Refine',refine);
fminopt = optimset('GradObj','on', 'Display', 'off','Hessian','on');
tfinal = 0;

% find MLE of unpenalized coefficients
setKeep = ~penidx;      % set of unpenalized coefficients
setPenZ = penidx;       % set of penalized coefficients that are zero
setPenNZ = false(p,1);  % set of penalized coefficients that are non-zero
coeff = zeros(p,1);     % subgradient coefficients
if (nnz(setKeep)>min(n,p))
    error('number of unpenalized coefficients exceeds rank of X');
end
if (any(setKeep))
    x0 = fminunc(@glmfun,zeros(nnz(setKeep),1),fminopt, ...
        X(:,setKeep),y,wt,model);
    beta_path(setKeep,1) = x0;
else
    beta_path(:,1) = 0;
end

% determine the maximum rho to start
[~,d1f] = glmfun(beta_path(:,1),X,y,wt,model);
[~,inext] = max(abs(d1f));
rho = glm_maxlambda(X(:,inext),X*beta_path(:,1),y,wt,pentype,penparam,model);
rho_path(1) = rho;

% determine active set and refine solution
rho = max(rho-tiny,0);
% update activeSet
x0 = glm_sparsereg(X,y,wt,rho,beta_path(:,1),penidx,maxrounds,...
    pentype,penparam,model);
setPenZ = abs(x0)<1e-8;
setPenNZ = ~setPenZ;
setPenZ(setKeep) = false;
setPenNZ(setKeep) = false;
setActive = setKeep|setPenNZ;
coeff(setPenNZ) = sign(x0(setPenNZ));
% improve parameter estimates
[x0, fval] = fminunc(@objfun, x0(setActive), fminopt, rho);
rho_path = [rho_path rho];
beta_path(setActive,end+1) = x0;
rho_kinks = length(rho_path);
fval_kinks = fval;

% main loop for path following
for k=2:maxiters

    % Solve ode until the next kink or discontinuity
    tstart = rho_path(end);
    [tseg,xseg] = ode45(@odefun,[tstart tfinal],x0,odeopt);

    % accumulate solution path
    rho_path = [rho_path tseg']; %#ok<*AGROW>
    beta_path(setActive,(end+1):(end+size(xseg,1))) = xseg';

    % update activeSet
    rho = max(rho_path(end)-tiny,0);
    x0 = beta_path(:,end);
    x0(setPenZ) = coeff(setPenZ);
    x0 = glm_sparsereg(X,y,wt,rho,x0,penidx,maxrounds,...
        pentype,penparam,model);
    setPenZ = abs(x0)<1e-8;
    setPenNZ = ~setPenZ;
    setPenZ(setKeep) = false;
    setPenNZ(setKeep) = false;
    setActive = setKeep|setPenNZ;
    coeff(setPenNZ) = sign(x0(setPenNZ));

    % improve parameter estimates
    [x0,fval] = fminunc(@objfun, x0(setActive), fminopt, rho);
    rho_path = [rho_path rho];
    beta_path(setActive,end+1) = x0;
    rho_kinks = [rho_kinks length(rho_path)];
    fval_kinks = [fval_kinks fval];
    tstart = rho;
    
    % termination
    if (tstart<=0 || nnz(setActive)>maxpreds)
        break;
    end
    if ( n<p && nnz(setActive)>=maxpreds)
        break;
    end
end
fval_kinks = fval_kinks + norm(wt.*y)^2;

    function [value,isterminal,direction] = events(t,x)
        value = ones(p,1);
        value(setPenNZ) = x(penidx(setActive));
        % try coordinate descent direction for zero coeffs
        if (any(setPenZ))
            xPenZ_trial = glm_thresholding(X(:,setPenZ), ...
                X(:,setActive)*x,y,wt,t,pentype,penparam,model);
            coeff(setPenZ) = xPenZ_trial;
            value(setPenZ) = abs(xPenZ_trial)==0;
        end
        isterminal = true(p,1);
        direction = zeros(p,1);
    end%EVENTS

    function dx = odefun(t, x)
        [~,~,d2pen,dpendrho] = ...
            penalty_function(x(penidx(setActive)),t,pentype,penparam);
        dx = zeros(length(x),1);
        if (any(setPenNZ))
            dx(penidx(setActive)) = dpendrho.*coeff(setPenNZ);
        end
        [~,~,M] = glmfun(x,X(:,setActive),y,wt,model);
        diagidx = find(penidx(setActive));
        diagidx = (diagidx-1)*length(x) + diagidx;
        M(diagidx) = M(diagidx) + d2pen;
        dx = - M\dx;
        if (any(isinf(dx)))
            dx(isinf(dx)) = 1e8*sign(dx(isinf(dx)));
        end
    end%ODEFUN

    function [f, d1f, d2f] = objfun(x, t)
        if (nargout<=1)
            [pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss] = glmfun(x,X(:,setActive),y,wt,model);
            f = loss + sum(pen);
        elseif (nargout==2)
            [pen,d1pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss,lossd1] = glmfun(x,X(:,setActive),y,wt,model);
            f = loss + sum(pen);
            d1f = lossd1;
            if (any(setPenNZ))
                d1f(penidx(setActive)) = d1f(penidx(setActive)) + ...
                    coeff(setPenNZ).*d1pen;
            end
        elseif (nargout==3)
            [pen,d1pen,d2pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss,lossd1,lossd2] = glmfun(x,X(:,setActive),y,wt,model);
            f = loss + sum(pen);
            d1f = lossd1;
            if (any(setPenNZ))
                d1f(penidx(setActive)) = d1f(penidx(setActive)) + ...
                    coeff(setPenNZ).*d1pen;
            end
            d2f = lossd2;
            diagidx = find(penidx(setActive));
            diagidx = (diagidx-1)*length(x) + diagidx;
            d2f(diagidx) = d2f(diagidx) + d2pen;
        end
    end%objfun

    function [loss,lossd1,lossd2] = glmfun(beta,X,y,wt,model)
        inner = X*beta;
        switch upper(model)
            case 'LOGISTIC'
                loss = - sum(wt.*(y.*inner-log(1+exp(inner))));
        end
        if (nargout>1)
           switch upper(model)
               case 'LOGISTIC'
                   prob = exp(inner)./(1+exp(inner));
                   lossd1 = - sum(bsxfun(@times, X, wt.*(y-prob)),1)';
           end
        end
        if (nargout>2)
            switch upper(model)
                case 'LOGISTIC'
                    lossd2 = X' * bsxfun(@times, wt.*prob.*(1-prob), X);
            end
        end
    end

end