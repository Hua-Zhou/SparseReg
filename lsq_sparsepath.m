function [rho_path,beta_path,eb_path,rho_kinks,fval_kinks] = ...
    lsq_sparsepath(X,y,varargin)
% LSQ_SPARSEPATH Solution path of sparse linear regression
%
%   [RHO_PATH,BETA_PATH] = LSQ_SPARSEPATH(X,Y) computes the solution path
%   of penalized linear regression using the predictor matrix X and
%   response Y. The result RHO_PATH holds rhos along the patha. The result
%   BETA_PATH holds solution vectors at each rho. By default it fits the
%   lasso regression.
%
%   [RHO_PATH,BETA_PATH] = LSQ_SPARSEPATH(X,Y,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit. Parameters are:
%
%       'maxpreds' - maximum number of top predictors requested.
%
%       'penalty' - ENET|LOG|MCP|POWER|SCAD
%
%       'penidx' - a logical vector indicating penalized coefficients.
%
%       'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%
%       'weights' - a vector of prior weights.
%
%   [RHO_PATH,BETA_PATH,EB_PATH,RHO_KINKS,FVAL_KINKS] = LSQ_SPARSEPATH(...)
%   returns the kinks of the solution paths and objective values at kinks
%
%   REFERENCE
%
%   EXAMPLE
%
%   See also LSQ_SPARSEREG,GLM_SPARSEREG,GLM_SPARSEPATH.

% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
[n,p] = size(X);
rankX = rank(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addParamValue('maxpreds', rankX, @(x) isnumeric(x) && x>0);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', 1, @isnumeric);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);

% parse inputs
argin.parse(X,y,varargin{:});
maxpreds = round(argin.Results.maxpreds);
penidx = reshape(argin.Results.penidx,p,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
wt = reshape(argin.Results.weights,n,1);
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

% precompute and allocate storage for path
tiny = 1e-4;
sum_x_squares = sum(bsxfun(@times,X.^2,wt),1);
islargep = maxpreds<p || p>=1000;
if (islargep)
    beta_path = sparse(p,1);
else
    beta_path = zeros(p,1);
end
rho_path = 0;

% find MLE of unpenalized coefficients
setKeep = ~penidx;      % set of unpenalized coefficients
setPenZ = penidx;       % set of penalized coefficients that are zero
setPenNZ = false(p,1);  % set of penalized coefficients that are non-zero
coeff = zeros(p,1);     % subgradient coefficients
setActive = setKeep;    % set of active coefficients
if (nnz(setKeep)>rankX)
    error('number of unpenalized coefficients exceeds rank of X');
end
XsetActive = X(:,setActive);
if (any(setActive))
    beta_path(setActive,1) = ...
        (XsetActive'*bsxfun(@times,XsetActive,wt))\(XsetActive'*(wt.*y));
else
    beta_path(:,1) = 0;
end

% set up ODE solver and unconstrained optimizer
maxiters = 3*rankX;         % max iterations for path algorithm
maxrounds = 3;              % max iterations for lsq_sparsereg
refine = 1;
odeopt = odeset('Events',@events,'Refine',refine);
% fminopt = optimset('GradObj','on', 'Display', 'off','Hessian','on');
tfinal = 0;

% determine the maximum rho to start
res = y-XsetActive*beta_path(setActive,1);
d1f = - ((wt.*res)'*X)';
[d1fnext,inext] = max(abs(d1f));
rho = lsq_maxlambda(sum_x_squares(inext),d1fnext,pentype,penparam);
rho_path(1) = rho;

% determine active set and refine solution
rho = max(rho-tiny,0);
% update activeSet
x0 = lsq_sparsereg(X,y,rho,'weights',wt,'x0',beta_path(:,1), ...
    'sum_x_squares',sum_x_squares,'penidx',penidx,'maxiter',maxrounds,...
    'penalty',pentype,'penparam',penparam);
setPenZ = abs(x0)<1e-8;
setPenNZ = ~setPenZ;
setPenZ(setKeep) = false;
setPenNZ(setKeep) = false;
setActive = setKeep|setPenNZ;
coeff(setPenNZ) = sign(x0(setPenNZ));
XsetActive = X(:,setActive);
xsetActive = x0(setActive);
M = XsetActive'*bsxfun(@times, XsetActive, wt); % quadratic part
diagidx = find(penidx(setActive));
diagidx = (diagidx-1)*nnz(setActive) + diagidx;
% accumulate kink information
rho_kinks = length(rho_path);
fval_kinks = objfun(xsetActive,rho);

% main loop for path following
for k=2:maxiters
    
    % Solve ode until the next kink or discontinuity
    tstart = rho_path(end);
    [tseg,xseg] = ode45(@odefun,[tstart tfinal],xsetActive,odeopt);
    
    % accumulate solution path
    rho_path = [rho_path tseg']; %#ok<*AGROW>
    beta_path(setActive,(end+1):(end+size(xseg,1))) = xseg';
    
    % update activeSet
    rho = max(rho_path(end)-tiny,0);
    x0 = beta_path(:,end);
    x0(setPenZ) = coeff(setPenZ);
    x0 = lsq_sparsereg(X,y,rho,'weights',wt,'x0',x0, ...
        'sum_x_squares',sum_x_squares,'penidx',penidx,'maxiter',maxrounds,...
        'penalty',pentype,'penparam',penparam);
    setPenZ = abs(x0)<1e-8;
    setPenNZ = ~setPenZ;
    setPenZ(setKeep) = false;
    setPenNZ(setKeep) = false;
    setActive = setKeep|setPenNZ;
    coeff(setPenNZ) = sign(x0(setPenNZ));
    XsetActive = X(:,setActive);
    xsetActive = x0(setActive);
    M = XsetActive'*bsxfun(@times, XsetActive, wt); % quadratic part
    diagidx = find(penidx(setActive));
    diagidx = (diagidx-1)*nnz(setActive) + diagidx;
    
    % accumulate kink information
    rho_kinks = [rho_kinks length(rho_path)];
    fval_kinks = [fval_kinks objfun(xsetActive,rho)];
    tstart = rho;
    
    % termination
    if (tstart<=0 || nnz(setActive)>maxpreds)
        break;
    end
    if ( n<p && nnz(setActive)>=maxpreds)
        break;
    end
end

% compute the emprical Bayes criterion along the path
if (strcmpi(pentype,'enet') && penparam==1)
    pentype = 'power';
end
compute_eb_path = strcmpi(pentype,'power') && nargin>=3;
if (compute_eb_path)
    eb_path = zeros(1,length(rho_path));
    for t=1:length(eb_path)
        setPenZ = abs(beta_path(:,t))<1e-8;
        setPenNZ = ~setPenZ;
        setPenZ(setKeep) = false;
        setPenNZ(setKeep) = false;
        setActive = setKeep|setPenNZ;
        XsetActive = X(:,setActive);
        [objf,~,objd2f] = objfun(beta_path(setActive,t),rho_path(t));
        q = nnz(setActive);
        if (strcmpi(pentype,'power'))
            if (rho_path(t)>0)
                eb_path(t) = - q*(.5*log(pi/2) + log(penparam) ...
                    + log(rho_path(t))/penparam - gammaln(1/penparam)) ...
                    + ((n-q)/2-q/penparam) ...
                    * (1 + log(objf) - log((n-q)/2+q/penparam)) ...
                    + log(det(objd2f))/2;
            else
                eb_path(t) = nan;
            end
        else
        end
    end
else
    eb_path = nan;
end

    function [value,isterminal,direction] = events(t,x)
        isterminal = true(p,1);
        direction = zeros(p,1);
        value = ones(p,1);
        res = y-XsetActive*x;
        lossd1 = -((wt.*res)'*X)';
        x_trial = lsq_thresholding(sum_x_squares,lossd1,t,pentype,penparam);
        value(penidx) = x_trial(penidx);
        coeff(setPenZ) = value(setPenZ);
        value(setPenZ) = abs(value(setPenZ))<eps;
    end%EVENTS

    function dx = odefun(t, x)
        [~,~,d2pen,dpendrho] = ...
            penalty_function(x(penidx(setActive)),t,pentype,penparam);
        dx = zeros(length(x),1);
        if (any(setPenNZ))
            dx(penidx(setActive)) = dpendrho.*coeff(setPenNZ);
        end
        H = M;
        H(diagidx) = H(diagidx)+d2pen;
        dx = - H\dx;
        if (any(isinf(dx)))
            dx(isinf(dx)) = 1e8*sign(dx(isinf(dx)));
        end
    end%ODEFUN

    function [f, d1f, d2f] = objfun(x, t)
        [pen,d1pen,d2pen] = ...
            penalty_function(x(penidx(setActive)),t,pentype,penparam);
        res = y-XsetActive*x;
        f = .5*sum(wt.*res.^2) + sum(pen);
        if (nargout>1)
            d1f = -  ((wt.*res)'*XsetActive)';
            if (any(setPenNZ))
                d1f(penidx(setActive)) = d1f(penidx(setActive)) + ...
                    coeff(setPenNZ).*d1pen;
            end
        end
        if (nargout>2)
            d2f = XsetActive'*bsxfun(@times, XsetActive, wt);
            diagidx = find(penidx(setActive));
            diagidx = (diagidx-1)*length(x) + diagidx;
            d2f(diagidx) = d2f(diagidx) + d2pen;
        end
    end%objfun

end