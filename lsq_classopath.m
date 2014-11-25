function [rhopath, betapath, dualpathEq, dualpathIneq] ...
    = lsq_classopath(X, y, A, b, Aeq, beq, varargin)
% LSQ_CLASSOPATH Constrained lasso solution path
%   BETAHAT = LSQ_SPARSEREG(X, y, lambda) fits penalized linear regression
%   using the predictor matrix X, response Y, and tuning parameter value
%   LAMBDA. The result BETAHAT is a vector of coefficient estimates. By
%   default it fits the lasso regression.
%
%   BETAHAT = LSQ_SPARSEREG(X,y,lambda,'PARAM1',val1,'PARAM2',val2,...)
%   allows you to specify optional parameter name/value pairs to control
%   the model fit.
%
% INPUT:
%   X - n-by-p design matrix
%   y - n-by-1 response vector
%   lambda - penalty tuning parameter
%   A - inequality constraint matrix
%   b - inequality constraint vector
%   Aeq - equality constraint matrix
%   beq - equality constraint vector
%
% OPTIONAL NAME-VALUE PAIRS:
%   'method' - 'cd' (default) or 'qp' (quadratic programming, only for
%      lasso)
%   'penidx' - a logical vector indicating penalized coefficients
%   'qp_solver' - 'matlab' (default), or 'GUROBI'
%
% OUTPUT:
%
% See also LSQ_SPARSEPATH,GLM_SPARSEREG,GLM_SPARSEPATH.
%
% Example
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
argin.addParamValue('direction', 'decrease', @ischar);
argin.addParamValue('qp_solver', 'matlab', @ischar);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);

% parse inputs
y = reshape(y,n,1);
argin.parse(X,y,A,b,Aeq,beq,varargin{:});
direction = argin.Results.direction;
qp_solver = argin.Results.qp_solver;
penidx = reshape(argin.Results.penidx,p,1);

% check validity of qp_solver
if ~(strcmpi(qp_solver, 'matlab') || strcmpi(qp_solver, 'GUROBI'))
    error('sparsereg:lsq_classopath:qp_solver', ...
        'qp_solver not recognied');
end

%%% start here for debugging %%%
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

% intialization
H = X'*X;
%%% end here %%%
if strcmpi(direction, 'increase')
    
    % initialize beta by quadratic programming
    if strcmpi(qp_solver, 'matlab')
        % use Matlab lsqlin
        options.Algorithm = 'interior-point-convex';
        options.Display = 'off';
        [x,~,~,~,~,lambda] = lsqlin(X,y,A,b,Aeq,beq,[],[],[],options);
        betapath(:,1) = x;
        dualpathEq(:,1) = lambda.eqlin;
        dualpathIneq(:,1) = lambda.ineqlin;
    elseif strcmpi(qp_solver, 'GUROBI')
        % use GUROBI solver if possible
        gmodel.obj = - X'*y;
        gmodel.A = sparse([Aeq; A]);
        gmodel.sense = [repmat('=', m1, 1); repmat('<', m2, 1)];
        gmodel.rhs = [beq; b];
        gmodel.Q = sparse(H)/2;
        gmodel.lb = - inf(p,1);
        gmodel.ub = inf(p,1);
        gmodel.objcon = norm(y)^2/2;
        gparam.OutputFlag = 0;
        gresult = gurobi(gmodel, gparam);
        betapath(:,1) = gresult.x;
        dualpathEq(:,1) = gresult.pi(m2+1:end);
        dualpathIneq(:,1) = reshape(gresult.pi(1:m2), m2, 1);
    end
    setActive = abs(betapath(:,1))>1e-16 | ~penidx;
    nActive = nnz(setActive);
    betapath(~setActive,1) = 0;
    setIneqBorder = dualpathIneq(:,1)>0;
    residIneq = A*betapath(:,1) - b;
    
    % intialize subgradient vector
    rhopath(1) = 0;
    subgrad = zeros(p,1);
    subgrad(setActive) = sign(betapath(setActive,1));
    subgrad(~penidx) = 0;

    % sign in path direction
    dirsgn = -1;
    % initialize k for manually looking at path following loop
    k = 2;
    
elseif strcmpi(direction, 'decrease')
    
    % initialize beta by linear programming
    if strcmpi(qp_solver, 'matlab')
        % use Matlab lsqlin
        [x,~,~,~,lambda] = ...
            linprog(ones(2*p,1),[A -A],b,[Aeq -Aeq],beq, ...
            zeros(2*p,1), inf(2*p,1));
        betapath(:,1) = x(1:p) - x(p+1:end);
        dualpathEq(:,1) = lambda.eqlin;
        dualpathIneq(:,1) = lambda.ineqlin;
    elseif strcmpi(qp_solver, 'GUROBI')
        % use GUROBI solver if possible
        gmodel.obj = ones(2*p,1);
        gmodel.A = sparse([A -A; Aeq -Aeq]);
        gmodel.sense = [repmat('<', m2, 1); repmat('=', m1, 1)];
        gmodel.rhs = [b; beq];
        gmodel.lb = zeros(2*p,1);
        gparam.OutputFlag = 0;
        gresult = gurobi(gmodel, gparam);
        betapath(:,1) = gresult.x(1:p) - gresult.x(p+1:end);
        dualpathEq(:,1) = gresult.pi(m2+1:end);
        dualpathIneq(:,1) = reshape(gresult.pi(1:m2), m2, 1);
    end
    
    % initialize sets
    setActive = abs(betapath(:,1))>1e-16 | ~penidx;
    betapath(~setActive,1) = 0;
    setIneqBorder = dualpathIneq(:,1)>0;
    residIneq = A*betapath(:,1) - b;

    % find the maximum rho and initialize subgradient vector
    resid = y - X*betapath(:,1);
    subgrad = X'*resid - Aeq'*dualpathEq(:,1) - A'*dualpathIneq(:,1);
    subgrad(setActive) = 0;
    [rhopath(1), idx] = max(abs(subgrad));
    subgrad(setActive) = sign(betapath(setActive,1));
    subgrad(~setActive) = subgrad(~setActive)/rhopath(1);
    setActive(idx) = true;
    nActive = nnz(setActive);
    
    % sign in path direction
    dirsgn = 1;
    % initialize k for manually looking at path following loop
    k = 2;
end

%k = 8;
% main loop for path following
s = warning('error', 'MATLAB:nearlySingularMatrix'); %#ok<CTPCT>
for k = 2:maxiters  %7 for simultaneity issue (when increasing)
    
    % path following direction
    M = [H(setActive, setActive) Aeq(:,setActive)' ...
        A(setIneqBorder,setActive)']; 
    M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ... 
        [Aeq(:,setActive); A(setIneqBorder,setActive)];
    try
%          dir = dirsgn ... 
%              * (M \ [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
        dir = dirsgn ...
            * (pinv(M) * ...
            [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
        
    catch
        break;
    end
    dirSubgrad = ...
         - [H(~setActive, setActive) Aeq(:,~setActive)' ...
         A(setIneqBorder,~setActive)'] * dir;
    dirResidIneq = A(~setIneqBorder,setActive)*dir(1:nActive);

%     % terminate path following
%     if max(abs(dir)) < 1e-8
%         break;
%     end            
    
    % next rho for beta
    nextrhoBeta = inf(p, 1);
    nextrhoBeta(setActive) = - betapath(setActive,k-1) ...
        ./ dir(1:nActive);
    
    % coefficient becoming positive 
    t1 = rhopath(k-1)*(1 - subgrad(~setActive)) ./ (dirSubgrad + dirsgn);
    %   t1 = rhopath(k-1)*(1 - subgrad(~setActive)) ./ ...
     %     (-dirSubgrad*dirsgn + dirsgn);
    %t1(t1<0) = inf; % hitting ceiling
    t1(t1<=0) = inf; % hitting ceiling
    t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
        ./ (dirSubgrad - dirsgn);
      %  t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
        %         ./ (-dirSubgrad*dirsgn - dirsgn);
      %  t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
      %   ./ (dirSubgrad + dirsgnb);
    % t2(t2<0) = inf; % hitting floor
    t2(t2<=0) = inf; % hitting floor
    %nextrhoBeta(~setActive) = min(t1, t2);
    nextrhoBeta(~setActive) = min(t1, t2);
    nextrhoBeta(nextrhoBeta<=1e-8 | ~penidx) = inf;
    
   %# begin: original code for subgradient
%    t1 = rhopath(k-1)*(1 - subgrad(~setActive)) ./ (dirSubgrad + dirsgn);
%    %   t1 = rhopath(k-1)*(1 - subgrad(~setActive)) ./ ...
%     %      (-dirSubgrad*dirsgn + dirsgn);
%     t1(t1<0) = inf; % hitting ceiling
%     t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
%         ./ (dirSubgrad - dirsgn);
% %    t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
% %             ./ (-dirSubgrad*dirsgn - dirsgn);
% %    t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
%  %      ./ (dirSubgrad + dirsgn);
%     t2(t2<0) = inf; % hitting floor
%     %nextrhoBeta(~setActive) = min(t1, t2);
%     nextrhoBeta(~setActive) = min(t1, t2);
%     nextrhoBeta(nextrhoBeta<=1e-8 | ~penidx) = inf;
   %# end: original code for subgradient 
   
   
    % next rho for inequality constraints
    nextrhoIneq = inf(m2, 1);
    % left off here:
    nextrhoIneq(setIneqBorder) = - dualpathIneq(setIneqBorder,k-1) ...
        ./ reshape(dir(nActive+m1+1:end), nnz(setIneqBorder),1);     
    nextrhoIneq(~setIneqBorder) = - residIneq(~setIneqBorder) ...
        ./ dirResidIneq;
    nextrhoIneq(nextrhoIneq<0) = inf;
    
    % determine next rho
    [chgrho,idx] = min([nextrhoBeta; nextrhoIneq]);
    
%     chgrho = min([nextrhoBeta; nextrhoIneq]);
%     % find indices corresponding to this chgho
%     idx = find(([nextrhoBeta; nextrhoIneq]-chgrho)<=1e-16); 
%        
    % terminate path following
    if isinf(chgrho)
        break;
    end
    
    % move to next rho
    if rhopath(k-1) - dirsgn*chgrho < 0 % may wanna change to maxrho
        chgrho = rhopath(k-1);              % for increasing direction?
    end
    rhopath(k) = rhopath(k-1) - dirsgn*chgrho;
    % this also doesn't make sense to me...but making the change breaks it
    betapath(setActive,k) = betapath(setActive,k-1) ...
         + chgrho*dir(1:nActive);
   % betapath(setActive,k) = betapath(setActive,k-1) ...
    %    - dirsgn*chgrho*dir(1:nActive);
    dualpathEq(:,k) = dualpathEq(:,k-1) ...
        + chgrho*reshape(dir(nActive+1:nActive+m1),m1,1);
    dualpathIneq(setIneqBorder,k) = dualpathIneq(setIneqBorder,k-1) ...
        + chgrho*reshape(dir(nActive+m1+1:end), nnz(setIneqBorder),1);  
     subgrad(~setActive) = ...
         (rhopath(k-1)*subgrad(~setActive) + chgrho*dirSubgrad)/rhopath(k);
   % subgrad(~setActive) = ...
    %    (rhopath(k-1)*subgrad(~setActive) - dirsgn*chgrho*dirSubgrad)...
     %   /rhopath(k);
    residIneq = A*betapath(:,k) - b;
    
    % update sets
    % what about >=2 variables become zero/nonzero together ???
    % old code:    
    if idx<=p && setActive(idx)
        % an active coefficient hits 0, or
        setActive(idx) = false;
    elseif idx<=p && ~setActive(idx)
        % a zero coefficient becomes nonzero
        setActive(idx) = true;
    elseif idx>p
        % an ineq on boundary becomes strict, or
        % a strict ineq hits boundary
        setIneqBorder(idx-p) = ~setIneqBorder(idx-p);
    end
    nActive = nnz(setActive);

% 
%         j=3;
%     for j = 1:length(idx)
%         curidx = idx(j);
%         if curidx<=p && setActive(curidx)
%             % an active coefficient hits 0, or
%             setActive(curidx) = false;
%         elseif curidx<=p && ~setActive(curidx)
%             % a zero coefficient becomes nonzero
%             setActive(curidx) = true;
%         elseif curidx>p
%             % an ineq on boundary becomes strict, or
%             % a strict ineq hits boundary
%             setIneqBorder(curidx-p) = ~setIneqBorder(curidx-p);
%         end
%     end
%     
%     setActive = abs(betapath(:,k))>1e-16 | ~penidx;
%     betapath(~setActive,k) = 0;
%     
     
    
    % determine new number of active coefficients
    %nActive = nnz(setActive);
end

% clean up
warning(s);
betapath(:, k:end) = [];
dualpathEq(:, k:end) = [];
dualpathIneq(:, k:end) = [];
rhopath(k:end) = [];

end