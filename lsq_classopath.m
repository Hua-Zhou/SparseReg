function [rhopath,betapath,dualpathEq,dualpathIneq] ...
    = lsq_classopath(X,y,A,b,Aeq,beq,varargin)
% LSQ_CLASSOPATH Constrained lasso solution path
%   BETAHAT = LSQ_SPARSEREG(X,y,lambda) fits penalized linear regression
%   using the predictor matrix X, response Y, and tuning parameter value
%   LAMBDA. The result BETAHAT is a vector of coefficient estimates. By
%   default it fits the lasso regression.
%
%   BETAHAT = LSQ_SPARSEREG(X,y,lambda,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit.
%
% INPUT:
%
%   X - n-by-p design matrix
%   y - n-by-1 response vector
%   lambda - penalty tuning parameter
%   A - inequality constraint matrix
%   b - inequality constraint vector
%   Aeq - equality constraint matrix
%   beq - equality constraint vector
%
% OPTIONAL NAME-VALUE PAIRS:
%
%   'method' - 'cd' (default) or 'qp' (quadratic programming, only for
%      lasso)
%   'penidx' - a logical vector indicating penalized coefficients
%   'qp_solver' - 'matlab' (default), or 'GUROBI'
%
% OUTPUT:
%
% See also LSQ_SPARSEPATH,GLM_SPARSEREG,GLM_SPARSEPATH.
%
% Eexample
%
% References
%

% Copyright 2014 North Carolina State University
% Hua Zhou (hua_zhou@ncsu.edu) and Brian Gaines

% input parsing rule
[n,p] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('A', @(x) size(x,2)==p || isempty(x));
argin.addRequired('b', @(x) isnumeric(x) || isempty(x));
argin.addRequired('Aeq', @(x) size(x,2)==p || isempty(x));
argin.addRequired('beq', @(x) isnumeric(x) || isempty(x));
argin.addParamValue('qp_solver', 'matlab', @ischar);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);

% parse inputs
y = reshape(y,n,1);
argin.parse(X,y,A,b,Aeq,beq,varargin{:});
qp_solver = argin.Results.qp_solver;
%penidx = reshape(argin.Results.penidx,p,1);

% check validity of qp_solver
if ~(strcmpi(qp_solver, 'matlab') || strcmpi(qp_solver, 'GUROBI'))
    error('sparsereg:lsq_classopath:qp_solver', ...
        'qp_solver not recognied');
end

% allocate space for path solution
m1 = size(Aeq, 1);  % # equality constraints
if isempty(Aeq)
    Aeq = zeros(0,p);
    beq = zeros(0,1);
end
m2 = size(A, 1);    % # inequality constraints
if isempty(A)
    A = zeros(0,p);
    b = zeros(0,1);
end
maxiters = 5*(p+m2);    % max number of path segments to consider
betapath = zeros(p, maxiters);
dualpathEq = zeros(m1, maxiters);
dualpathIneq = zeros(m2, maxiters);
rhopath = zeros(1, maxiters);

% intialize beta by linear programming
if strcmpi(qp_solver, 'matlab')
    % use Matlab linprog
    [x,~,~,~,lambda] = ...
        linprog(ones(2*p,1),[A -A],b,[Aeq -Aeq],beq, ...
        zeros(2*p,1), inf(2*p,1));
    betapath(:,1) = x(1:p) - x(p+1:end);
    dualpathEq(:,1) = lambda.eqlin;
    dualpathIneq(:,1) = lambda.ineqlin;
elseif strcmpi(qp_solver, 'GUROBI')
    % use QUROBI solver if possible
    gmodel.obj = ones(2*p,1);
    gmodel.A = sparse([A -A; Aeq -Aeq]);
    gmodel.sense = [repmat('<', m1, 1); repmat('=', m2, 1)];
    gmodel.rhs = [b; beq];
    gmodel.lb = zeros(2*p,1);
    gparam.OutputFlag = 0;
    gresult = gurobi(gmodel, gparam);
    betapath(:,1) = gresult.x(1:p) - gresult.x(p+1:end);
    dualpathEq(:,1) = gresult.pi(1:m1);
    dualpathIneq(:,1) = reshape(gresult.pi(m1+1:end),m2,1);
end
setActive = abs(betapath(:,1))>1e-16;
betapath(~setActive,1) = 0;
setIneqBorder = dualpathIneq(:,1)>0;
residIneq = A*betapath(:,1) - b;

% find the maximum rho
resid = y - X(:,setActive)*betapath(setActive,1);
grad = X(:,~setActive)'*resid ...
    - Aeq(:,~setActive)'*dualpathEq(:,1) ...
    - A(:,~setActive)'*dualpathIneq(:,1);
rhopath(1) = max(abs(grad));

% intialize subgradient vector
subgrad = zeros(p,1);
subgrad(betapath(:,1)>0) = 1;
subgrad(betapath(:,1)<0) = -1;
subgrad(~setActive) = grad / rhopath(1);
setActive = abs(subgrad-1)<1e-8 | abs(subgrad+1)<1e-8;

% main loop for path following
H = X'*X;
for k = 2:maxiters
    
    % path following direction
    M = [H(setActive, setActive) Aeq(:,setActive)' A(:,setActive)'];
    M(end+1:end+m1+nnz(setIneqBorder), 1:nnz(setActive)) = ...
        [Aeq(:,setActive); A(:,setActive)];
    dir = M \ [subgrad(setActive); zeros(m1+nnz(setIneqBorder,1))];
    dirSubgrad = ...
        - [H(~setActive, setActive), Aeq(:,~setActive)' A(:,~setActive)']...
        * dir;
    dirResidIneq = A(~setIneqBorder,~setActive)'*dir(1:nnz(setActive));
    
    % next rho for beta
    nextrhoBeta = inf(p, 1);
    nextrhoBeta(setActive) = - betapath(setActive,k-1) ...
        ./ dir(1:nnz(setActive));
    nextrhoBeta(~setActive) = max( ...
        rhopath(:,k-1)*(subgrad(~setActive)+1)./(-dirSubgrad+1), ...
        rhopath(:,k-1)*(subgrad(~setActive)-1)./(-dirSubgrad-1) );
    nextrhoBeta(nextrhoBeta<0) = inf;
    
    % next rho for inequality constraints
    nextrhoIneq = inf(m2, 1);
    nextrhoIneq(setIneqBorder) = - dualpathIneq(setIneqBorder,k-1) ...
        ./ dir(end-nnz(setIneqBorder)+1,end);
    nextrhoIneq(~setIneqBorder) = - residIneq ./ dirResidIneq;
    nextrhoIneq(nextrhoIneq<0) = inf;
    
    % determine next rho
    [chgrho, idx] = min([nextrhoBeta; nextrhoIneq]);
    
    % move to next rho
    rhopath(k) = rhopath(k-1) - chgrho;
    betapath(setActive,k) = betapath(setActive,k-1) ...
        + chgrho*dir(1:nnz(setActive));
    dualpathEq(:,k) = dualpathEq(:,k-1) ...
        + chgrho*dir(nnz(setActive)+1:nnz(setActive)+m1);
    dualpathIneq(setIneqBorder,k) = dualpathIneq(setIneqBorder,k-1) ...
        + chgrho*dir(nnz(setActive)+m1+1:end);
    subgrad(~setActive) = ...
        (rhopath(k-1)*subgrad(~setActive) + chgtol*dirSubgrad)/rhopath(k);
    residIneq = A*betapath(:,k) - b;
    
    % update sets
    % what about >=2 variables become zero/nonzero together ???
    if idx<=p && setActive(idx)
        % an active coefficient hits 0
        setActive(idx) = false;
    elseif idx<=p && ~setActive(idx)
        % a zero coefficient becomes nonzero
        setActive(idx) = true;
    elseif idx>p && setIneqBorder(idx-p)
        % an ineq on boundary becomes strict
        setIneqBorder(idx-p) = false;
    elseif idx>p && ~setIneqBorder(idx-p)
        % a strict ineq hits boundary
        setIneqBorder(idx-p) = true;
    end

end

end