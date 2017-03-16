function [rho_path,beta_path,eb_path,rho_kinks,fval_kinks] = ...
    glm_sparsepath(X,y,model,varargin)
% GLM_SPARSEPATH Solution path of sparse GLM regression
%   [RHO_PATH,BETA_PATH] = GLM_SPARSEPATH(X,Y) computes the solution path
%   of penalized GLM regression using the predictor matrix X and response
%   Y. MODEL specifies the model: 'logistic' or 'loglinear'. The result
%   RHO_PATH holds rhos along the patha. The result BETA_PATH holds
%   solution vectors at each rho. By default it fits the lasso regression.
%
%   [RHO_PATH,BETA_PATH] = GLM_SPARSEPATH(X,Y,MDOEL,'PARAM1',val1,'PARAM2',val2,...) allows you to
%   specify optional parameter name/value pairs to control the model fit.
%   Parameters are:
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
%   [RHO_PATH,BETA_PATH,RHO_KINKS,FVAL_KINKS] = LSQ_SPARSEPATH(...) returns
%   the kinks of the solution paths and objective values at kinks
%
%   See also LSQ_SPARSEREG,GLM_SPARSEREG,GLM_SPARSEPATH.
%
%   References:
%

% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% input parsing rule
[n,p] = size(X);
rankX = rank(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('model', @(x) strcmpi(x,'logistic')||strcmpi(x,'loglinear'));
argin.addParameter('maxpreds', rankX, @(x) isnumeric(x) && x>0);
argin.addParameter('penalty', 'enet', @ischar);
argin.addParameter('penparam', 1, @isnumeric);
argin.addParameter('penidx', true(p,1), @(x) islogical(x) && length(x)==p);
argin.addParameter('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);

% parse inputs
y = reshape(y,n,1);
argin.parse(X,y,model,varargin{:});
maxpreds = round(argin.Results.maxpreds);
penidx = reshape(argin.Results.penidx,p,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
wt = reshape(argin.Results.weights,n,1);

if (strcmp(pentype,'ENET'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<1 || penparam>2)
        error('index parameter for ENET penalty should be in [1,2]');
    end
    isconvex = true;
elseif (strcmp(pentype,'LOG'))
    if (isempty(penparam))
        penparam = 1;
    elseif (penparam<0)
        error('index parameter for LOG penalty should be nonnegative');
    end
    isconvex = false;
elseif (strcmp(pentype,'MCP'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<=0)
        error('index parameter for MCP penalty should be positive');
    end
    isconvex = false;
elseif (strcmp(pentype,'POWER'))
    if (isempty(penparam))
        penparam = 1;   % lasso by default
    elseif (penparam<=0 || penparam>2)
        error('index parameter for POWER penalty should be in (0,2]');
    end
    if (penparam<1)
        isconvex  = false;
    else
        isconvex = true;
    end
elseif (strcmp(pentype,'SCAD'))
    if (isempty(penparam))
        penparam = 3.7;
    elseif (penparam<=2)
        error('index parameter for SCAD penalty should be larger than 2');
    end
    isconvex = false;
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

% precompute and allocate storage for path
X2 = X.^2;
tiny = 1e-4;
islargep = p>=1000;
if (islargep)
    beta_path = sparse(p,1);
else
    beta_path = zeros(p,1);
end
rho_path = 0;
eb_path = nan;

% set up ODE solver and unconstrained optimizer
maxiters = 2*rankX;         % max iterations for path algorithm
maxrounds = 1;              % max iterations for glm_sparsereg
refine = 1;
odeopt = odeset('Events',@events, 'Refine',refine);
fminopt = optimset('GradObj','on', 'Display', 'off','Hessian','on', ...
    'Algorithm','trust-region');
tfinal = 0;

% find MLE of unpenalized coefficients
setKeep = ~penidx;          % set of unpenalized coefficients
setPenZ = penidx;           % set of penalized coefficients that are zero
setPenNZ = false(p,1);      % set of penalized coefficients that are non-zero
setActive = setKeep;
coeff = zeros(p,1);         % subgradient coefficients
if (nnz(setKeep)>rankX)
    error('number of unpenalized coefficients exceeds rank of X');
end
if (any(setKeep))
    x0 = fminunc(@objfun,zeros(nnz(setKeep),1),fminopt,0);
    beta_path(setKeep,1) = x0;
    inner = X(:,setKeep)*x0;
else
    beta_path(:,1) = 0;
    inner = zeros(n,1);
end

% determine the maximum rho to start
[~,d1f] = glmfun(inner,X,y,wt,model);
[~,inext] = max(abs(d1f));
rho = glm_maxlambda(X(:,inext),y,model,'weights',wt,'penalty',pentype, ...
    'penparam',penparam,'offset',X(:,setKeep)*beta_path(setKeep,1));
if (isnan(rho))
    warning('glm_sparsepath:nan', 'NaN encountered from glm_maxlambda');
    return;
else
    rho_path(1) = rho;
end

% determine active set and refine solution
rho = max(rho-tiny,0);
% update activeSet
x0 = glm_sparsereg(X,y,rho,model,'weights',wt,'x0',beta_path(:,1), ...
    'penidx',penidx,'maxiter',maxrounds,'penalty',pentype,'penparam',penparam);
if (any(isnan(x0)))
    warning('glm_sparsepath:nan', 'NaN encountered from glm_sparsereg');
    return;
end
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
    if (rho==0)
        break;
    end
    x0 = beta_path(:,end);
    if (~isconvex)
        x0(setPenZ) = coeff(setPenZ);
    end
    x0 = glm_sparsereg(X,y,rho,model,'weights',wt,'x0',x0,'penidx',penidx, ...
        'maxiter',maxrounds,'penalty',pentype,'penparam',penparam);
    if (any(isnan(x0)))
        warning('glm_sparsepath:nan', 'NaN encountered from glm_sparsereg');
        break;
    end
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
    if (n<p && nnz(setActive)>=maxpreds)
        break;
    end
    % detect separation in logistic and loglinear models
    inner = X(:,setActive)*x0;
    if (detect_separation(inner,y,model))
        warning('glm_sparsepath:logistic:separation',['separation detected; ' ...
            'perfect prediction achieved']);
        break;
    end
end

% compute the emprical Bayes criterion along the path
if (strcmpi(pentype,'enet') && penparam==1)
    pentype = 'power';
end
compute_eb_path = (nargin>=3 && ...
    (strcmpi(pentype,'power') || strcmpi(pentype,'log')));
if (compute_eb_path)
    eb_path = zeros(1,length(rho_path));
    for t=1:length(eb_path)
        setPenZ = abs(beta_path(:,t))<1e-8;
        setPenNZ = ~setPenZ;
        setPenZ(setKeep) = false;
        setPenNZ(setKeep) = false;
        setActive = setKeep|setPenNZ;
        [objf,~,objd2f] = objfun(beta_path(setActive,t), rho_path(t));
        if (strcmpi(pentype,'power'))
            if (rho_path(t)>0)
                eb_path(t) = - nnz(setActive)*(.5*log(pi/2) + log(penparam) ...
                    + log(rho_path(t))/penparam - gammaln(1/penparam)) ...
                    + objf + 0.5*real(log(det(objd2f)));
            else
                eb_path(t) = nan;
            end
        elseif (strcmpi(pentype,'log'))
            if (rho_path(t)<=1)
                eb_path(t) = nan;
            else
                eb_path(t) = -nnz(setActive)*(0.5*log(pi/2) ...
                    + log(rho_path(t)-1) ...
                    + (rho_path(t)-1)*log(penparam)) ...
                    + objf + 0.5*real(log(det(objd2f)));
            end
        end
    end
end

    function [value,isterminal,direction] = events(t,x)
        isterminal = true(p,1);
        direction = zeros(p,1);
        value = ones(p,1);
        value(setPenNZ) = x(penidx(setActive));
        inner = X(:,setActive)*x;
        % detect nan and complete separation
        if (any(isnan(x)) || detect_separation(inner,y,model))
            value(1) = 0;
            return;
        end
        if (isconvex)
            [~,lossD1PenZ] = glmfun(inner,X,y,wt,model);
            [~,penD1PenZ] = penalty_function(0,t,pentype,penparam);
            coeff(setPenZ) = 0;
            value(setPenZ) = abs(lossD1PenZ(setPenZ))<abs(penD1PenZ);
        elseif (any(setPenZ))
            % try thresholding for zero coeffs using
            % weighted least squares approximation
            [~,d1PenZ] = glmfun(inner,X,y,wt,model);
            glmwts = glmweights(inner,wt,model);
            d2PenZ = glmwts'*X2;
            xPenZ_trial = lsq_thresholding(d2PenZ(setPenZ),...
                d1PenZ(setPenZ),t,pentype,penparam);
            if (any(isnan(xPenZ_trial)))
                warning('glm_sparsepath:nan', ...
                    'NaN encountered from lsq_thresholding');
                return;
            end
            coeff(setPenZ) = xPenZ_trial;
            value(setPenZ) = abs(xPenZ_trial)<1e-8;
        end
    end%EVENTS

    function dx = odefun(t, x)
        inner = X(:,setActive)*x;
        [~,~,d2pen,dpendrho] = ...
            penalty_function(x(penidx(setActive)),t,pentype,penparam);
        dx = zeros(length(x),1);
        if (any(setPenNZ))
            dx(penidx(setActive)) = dpendrho.*coeff(setPenNZ);
        end
        [~,~,M] = glmfun(inner,X(:,setActive),y,wt,model);
        diagidx = find(penidx(setActive));
        diagidx = (diagidx-1)*length(x) + diagidx;
        M(diagidx) = M(diagidx) + d2pen;
        if (any(isnan(M(:))))
            warning('glm_sparsepath:nan', ...
                'NaN encountered from glm_sparsereg');
            return;
        end
        dx = - M\dx;
        if (any(isinf(dx)))
            dx(isinf(dx)) = 1e8*sign(dx(isinf(dx)));
        end
    end%ODEFUN

    function [f, d1f, d2f] = objfun(x, t)
        inner = X(:,setActive)*x;
        if (nargout<=1)
            [pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss] = glmfun(inner,X(:,setActive),y,wt,model);
            f = loss + sum(pen);
        elseif (nargout==2)
            [pen,d1pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss,lossd1] = glmfun(inner,X(:,setActive),y,wt,model);
            f = loss + sum(pen);
            d1f = lossd1;
            if (any(setPenNZ))
                d1f(penidx(setActive)) = d1f(penidx(setActive)) + ...
                    coeff(setPenNZ).*d1pen;
            end
        elseif (nargout==3)
            [pen,d1pen,d2pen] = ...
                penalty_function(x(penidx(setActive)),t,pentype,penparam);
            [loss,lossd1,lossd2] = glmfun(inner,X(:,setActive),y,wt,model);
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
    end%OBJFUN

    function [loss,lossd1,lossd2] = glmfun(inner,X,y,wt,model)
        big = 20;
        switch upper(model)
            case 'LOGISTIC'
                expinner = exp(inner);
                logterm = log(1+expinner);
                logterm(inner>big) = inner(inner>big);
                logterm(inner<-big) = 0;
                loss = - sum(wt.*(y.*inner-logterm));
            case 'LOGLINEAR'
                expinner = exp(inner);
                loss = - sum(wt.*(y.*inner-expinner)) + sum(gammaln(y+1));
        end
        if (nargout>1)
            switch upper(model)
                case 'LOGISTIC'
                    prob = expinner./(1+expinner);
                    prob(inner>big) = 1;
                    prob(inner<-big) = 0;
                    lossd1 = - ((wt.*(y-prob))'*X)';
                case 'LOGLINEAR'
                    lossd1 = - ((wt.*(y-expinner))'*X)';
            end
        end
        if (nargout>2)
            switch upper(model)
                case 'LOGISTIC'
                    lossd2 = X'*bsxfun(@times, wt.*prob.*(1-prob), X);
                case 'LOGLINEAR'
                    lossd2 = X'*bsxfun(@times, wt.*expinner, X);
            end
        end
    end%GLMFUN

    function [glmwts] = glmweights(inner,wt,model)
        big = 20;
        switch upper(model)
            case 'LOGISTIC'
                expinner = exp(inner);
                prob = expinner./(1+expinner);
                prob(inner>big) = 1;
                prob(inner<-big) = 0;
                glmwts = wt.*prob.*(1-prob);
            case 'LOGLINEAR'
                expinner = exp(inner);
                glmwts = wt.*expinner;
        end
    end%GLMWEIGHTS

    function s = detect_separation(inner,y,model)
        s = 0;  % 0 - no separation
        if (strcmpi(model,'logistic'))
            if (all(inner(y>0.5)>0) && all(inner(y<0.5)<0))
                s=1;
            end
        elseif (strcmpi(model,'loglinear'))
            if (all(inner(y<=eps)<=0) && all(abs(inner(y>eps))<eps))
                s=1;
            end
        end
    end%DETECT_SEPARATION

end