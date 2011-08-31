function [rho_path,beta_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,wt,penidx,maxpreds,penname,penargs)
% LSQ_SPARSEREG: 
%   Solution path for sparse least squares regression
%       .5*sum(wt.*(y-X*beta).^2)+rho*sum(penfun(beta(penidx)))
%
% Arguments:
%   X - n-by-p design matrix
%   y - n-by-1 responses
%   wt - n-by-1 weights; wt = ones(p,1) by default
%   penidx - p-by-1 logical index of the coefficients being penalized;
%       penalizing all coefficients by default
%   maxpreds - maximum number of top predictors requested; min(n,p) by
%       default
%   penname - 'enet'|'log'|'scad'|'mcp'|'bridge'
%   penargs - optional arguments for penalty function penname
%
% Output:
%   rho_path - rhos along the path
%   x_path - solution vectors at each rho
%   rho_kinks - kinks of the solution paths
%   fval_kinks - objective values at kinks
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou, hua_zhou@ncsu.edu
% RELEASE DATE: ??/??/????

% check arguments
[n,p] = size(X);
if (size(y,1)~=n)
    error('dimension of y does not match size(X,1)');
end
if (isempty(wt))
    wt = ones(n,1);
end
if (size(wt,1)~=n)
    error('dimension of wt does not match size(X,1)');
end
if (isempty(penidx))
    penidx = true(p,1); % penalize all coefficients by default
end
if (size(penidx,1)~=p)
    error('dimension of penidx does not match size(X,2)');
end
if (isempty(maxpreds) || maxpreds>=min(n,p))
    maxpreds = min(n,p);
end
if (isempty(penname))
    error('please specify the penalty: ''enet''|''log''|''scad''|''mcp''|''bridge''');
end
if (strcmpi(penname,'bridge'))
    PENFUN = @bridge_penalty;
elseif (strcmpi(penname,'enet'))
    PENFUN = @enet_penalty;
elseif (strcmpi(penname,'log'))
    PENFUN = @log_penalty;
elseif (strcmpi(penname,'mcp'))
    PENFUN = @mcp_penalty;
elseif (strcmpi(penname,'scad'))
    PENFUN = @scad_penalty;
end

% precompute and allocate storage for path
tiny = 1e-4;
b = - X'*(wt.*y);               % p-by-1
predl2 = sum(bsxfun(@times,X.^2,wt),1)';  % p-by-1, predictor l2 norms
largep = p>1e4;
if (largep)
    A = sparse(p,p);    % compute A on-the-fly
    beta_path = sparse(p,1);
else
    A = X'*bsxfun(@times,X,wt);    % p-by-p
    beta_path = zeros(p,1);
end
rho_path = 0;

% find MLE of unpenalized coefficients
setKeep = ~penidx;      % set of unpenalized coefficients
setPenZ = penidx;       % set of penalized coefficients that are zero
setPenNZ = false(p,1);  % set of penalized coefficients that are non-zero
coeff = zeros(p,1);
if (nnz(setKeep)>min(n,p))
    error('number of unpenalized coefficients exceeds rank of X!');
end
if (largep)
    A(:,setKeep) = X'*bsxfun(@times,X(:,setKeep),wt);
    A(setKeep,:) = A(:,setKeep)';
    setAcomputed = setKeep;
end
beta_path(setKeep,1) = - A(setKeep,setKeep)\b(setKeep);

% set up ODE solver
maxiters = 2*min([n,p]); % max iterations for path algorithm
maxrounds = 2;  % max iterations for optimizer OPTALGO
refine = 1;
odeopt = odeset('Events',@events, 'Refine',refine);
fminopt = optimset('GradObj','on', 'Display', 'off','Hessian','on');
tfinal = 0;

% determine the maximum rho to start
d1f = A(:,setKeep)*beta_path(setKeep,1)+b;
[d1fnext,inext] = max(abs(d1f));
rho = PENFUN([predl2(inext) d1fnext],inf,penargs);
rho_path(1) = rho;

% determin active set and refine solution
rho = max(rho-tiny,0);
% update activeSet
% x0 = cdesc_wrapper_xyformat(OPTALGO,A,b,rho,beta_path(:,end),...
%     penidx,maxrounds,penargs);
x0 = cdesc_wrapper(OPTALGO,X,y,wt,rho,beta_path(:,end),...
    penidx,maxrounds,penargs);
setPenZ = abs(x0)<1e-9;
setPenNZ = ~setPenZ;
setPenZ(setKeep) = false;
setPenNZ(setKeep) = false;
setActive = setKeep|setPenNZ;
coeff(setPenNZ) = sign(x0(setPenNZ));
if (largep)
    setAnew = find(setActive&(~setAcomputed));
    A(:,setAnew) = X'*bsxfun(@times,X(:,setAnew),wt);
    A(setAnew,:) = A(:,setAnew)';
    setAcomputed = setActive|setAcomputed;
end
% improve parameter estimates
[x0, fval] = fminunc(@objfun, x0(setActive), fminopt, rho);
rho_path = [rho_path rho];
beta_path(setActive,end+1) = x0;
rho_kinks = length(rho_path);
fval_kinks = fval;

% main loop for path following
for k=2:maxiters

%     display(k);
    % Solve ode until the next kink or discontinuity
    tstart = rho_path(end);
    [tseg,xseg] = ode15s(@odefun,[tstart tfinal],x0,odeopt);

    % accumulate solution path
    rho_path = [rho_path tseg']; %#ok<*AGROW>
    beta_path(setActive,(end+1):(end+size(xseg,1))) = xseg';

    % set up options for next round of ode
    % update activeSet
    rho = max(rho_path(end)-tiny,0);
    x0 = beta_path(:,end);
    x0(setPenZ) = coeff(setPenZ);
%     x0 = cdesc_wrapper_xyformat(OPTALGO,A,b,rho,x0,penidx,maxrounds,penargs);
    x0 = cdesc_wrapper(OPTALGO,X,y,wt,rho,x0,penidx,maxrounds,penargs);
    setPenZ = abs(x0)<1e-9;
    setPenNZ = ~setPenZ;
    setPenZ(setKeep) = false;
    setPenNZ(setKeep) = false;
    setActive = setKeep|setPenNZ;
    coeff(setPenNZ) = sign(x0(setPenNZ));
    if (largep)
        setAnew = find(setActive&(~setAcomputed));
        A(:,setAnew) = X'*bsxfun(@times,X(:,setAnew),wt);  %#ok<SPRIX>
        A(setAnew,:) = A(:,setAnew)'; %#ok<SPRIX>
        setAcomputed = setActive|setAcomputed;
    end
    % improve parameter estimates
    [x0,fval] = fminunc(@objfun, x0(setActive), fminopt, rho);
%     display(fval);
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
            d2PenZ = predl2(setPenZ);
            d1PenZ = A(setPenZ,setActive)*x+b(setPenZ);
            xPenz_trial = OPTALGO(d2PenZ,d1PenZ,t,penargs);
            value(setPenZ) = abs(xPenz_trial)==0;
            coeff(setPenZ) = xPenz_trial;
        end
        isterminal = true(p,1);
        direction = zeros(p,1);
    end%EVENTS

    function dx = odefun(t, x)
        [~,~,d2pen,dpendrho] = PENFUN(x(penidx(setActive)),t,penargs);
        dx = zeros(length(x),1);
        dx(penidx(setActive)) = dpendrho.*coeff(setPenNZ);
        M = A(setActive,setActive);
        diagidx = find(penidx(setActive));
        diagidx = (diagidx-1)*length(x) + diagidx;
        M(diagidx) = M(diagidx) + d2pen;
        dx = - M\dx;
        if (any(isinf(dx)))
            dx(isinf(dx)) = 1e12*sign(dx(isinf(dx)));
        end
    end%ODEFUN

    function [f, d1f, d2f] = objfun(x, t)
        [pen,d1pen,d2pen] = PENFUN(x(penidx(setActive)),t,penargs);
        f = 0.5*x'*A(setActive,setActive)*x + b(setActive)'*x ...
            + sum(pen);
        if (nargout>1)
            d1f = A(setActive,setActive)*x + b(setActive);
            d1f(penidx(setActive)) = d1f(penidx(setActive)) + ...
                coeff(setPenNZ).*d1pen;
        end
        if (nargout>2)
            d2f = A(setActive,setActive);
            diagidx = find(penidx(setActive));
            diagidx = (diagidx-1)*length(x) + diagidx;
            d2f(diagidx) = d2f(diagidx) + d2pen;
        end
    end%objfun

end