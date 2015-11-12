function [rhopath, betapath, dfPath, objValPath, stationarityConditionsPath, ...
    constraintsSatisfied, subgradientPath, violationsPath, dualpathEq, ...
    dualpathIneq] = lsq_classopath(X, y, A, b, Aeq, beq, varargin)
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
%   objValPath - value of the objective function along the solution path
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
% temp code for option of choosing method with multiple coefficients or not
argin.addParamValue('multCoeff', 'false', @ischar);
% temp code for changing the tolerance level for picking delta rho (chgrho)
argin.addParamValue('deltaRhoTol', 1e-8, @isnumeric);

% parse inputs
y = reshape(y, n, 1);
argin.parse(X, y, A, b, Aeq, beq, varargin{:});
direction = argin.Results.direction;
qp_solver = argin.Results.qp_solver;
multCoeff = argin.Results.multCoeff;
deltaRhoTol = argin.Results.deltaRhoTol;
penidx = reshape(argin.Results.penidx,p,1);

% check validity of qp_solver
if ~(strcmpi(qp_solver, 'matlab') || strcmpi(qp_solver, 'GUROBI'))
    error('sparsereg:lsq_classopath:qp_solver', ...
        'qp_solver not recognized');
end

%%% start here for (manual) debugging %%%
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
dfPath = Inf(2, maxiters);
objValPath = zeros(1, maxiters);
stationarityConditionsPath.values = zeros(p, maxiters);
stationarityConditionsPath.satisfied = Inf(1, maxiters);
constraintsSatisfied.eq = Inf(1, maxiters);
constraintsSatisfied.ineq = Inf(1, maxiters);
subgradientPath.values = zeros(p, maxiters);
subgradientPath.satisfied = Inf(1, maxiters);
subgradientPath.dir = NaN(p, maxiters);
subgradientPath.inactives = NaN(p, maxiters);
subgradientPath.rhoSubgrad = NaN(p, maxiters);
violationsPath = Inf(1, maxiters);


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
        dualpathEq(:,1) = gresult.pi(1:m1);
        dualpathIneq(:,1) = reshape(gresult.pi(m1+1:end), m2, 1);
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

    %%% store various things for debugging %%%
    % calculate value for objective function
    objValPath(1) = norm(y-X*betapath(:,1))^2/2 + ...
        rhopath(1)*sum(abs(betapath(:,1)));
    
    % calculate degrees of freedom (using two different methods, I believe
    % method 1 is more correct).  %Also, df are thresholded at zero.  
    rankAeq = rank(Aeq);
    dfPath(1, 1) = rank(X(:,  setActive)) - rankAeq;
    dfPath(2, 1) = nActive - rankAeq;
    %dfPath(1, 1) = max(rank(X(:,  setActive)) - rankAeq, 0);
    %dfPath(2, 1) = max(nActive - rankAeq, 0);
    
    % calculate the stationarity condition value
    stationarityConditionsPath.values(:, 1) = -X'*(y - X*betapath(:, 1)) + ...
        rhopath(1)*subgrad + Aeq'*dualpathEq(:, 1) + A'*dualpathIneq(:, 1);
    % see if stationarity condition is satisified
    stationarityConditionsPath.satisfied(1) = ...
        sum(abs(stationarityConditionsPath.values(:, 1)) < 1e-8) == p;
     
    % store subgradient
    subgradientPath.values(:, 1) = subgrad;
    % check that subgradient condition is satisfied
    subgradientPath.satisfied(1) = ...
        nnz(subgradientPath.values(:, 1) <= 1 & ...
        subgradientPath.values(:, 1) >= -1) == p;
    % indices for inactive coefficients (dirSubgrad entries)
    subgradientPath.inactives(1:size(find(setActive == 0)), 1) = ...
        find(setActive == 0);
    % calculate rho*subgrad
    subgradientPath.rhoSubgrad(:, 1) = rhopath(1)*subgrad;
   
    % set initial violations counter to 0
    violationsPath(1) = 0;
    
    % sign in path direction
    % dirsgn = -1;
    dirsgn = 1;
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
    
    % calculate value for objective function
    objValPath(1) = norm(y-X*betapath(:,1))^2/2 + ...
        rhopath(1)*sum(abs(betapath(:,1)));
    % calculate degrees of freedom (using two different methods, I believe
    % method 1 is more correct).  %Also, df are thresholded at zero.  
    rankAeq = rank(Aeq);
    dfPath(1, 1) = rank(X(:,  setActive)) - rankAeq;
    dfPath(2, 1) = nActive - rankAeq;
    %dfPath(1, 1) = max(rank(X(:,  setActive)) - rankAeq, 0);
    %dfPath(2, 1) = max(nActive - rankAeq, 0);
    
    % calculate the stationarity condition value
    stationarityConditionsPath.values(:, 1) = -X'*(y - X*betapath(:, 1)) + ...
        rhopath(1)*subgrad + Aeq'*dualpathEq(:, 1) + A'*dualpathIneq(:, 1);
    % see if stationarity condition is satisified
    stationarityConditionsPath.satisfied(1) = ...
        sum(abs(stationarityConditionsPath.values(:, 1)) < 1e-8) == p;
    % check if constraints are violated or not
    % equality
    if m1==0
        constraintsSatisfied.eq(1) = NaN;
    else
        constraintsSatisfied.eq(1) = ...
            sum(abs(Aeq*betapath(:, 1) - beq) < 1e-10) == m1;
    end
    % inequality
    if m2==0
        constraintsSatisfied.ineq(1) = NaN;
    else
        constraintsSatisfied.ineq(1) = ...
            sum(A*betapath(:, 1) - b < 1e-10) == m2;
    end
    
    % store subgradient
    subgradientPath.values(:, 1) = subgrad;
    % check that subgradient condition is satisfied
    subgradientPath.satisfied(1) = ...
        nnz(subgradientPath.values(:, 1) <= 1 & ...
        subgradientPath.values(:, 1) >= -1) == p;
    % indices for inactive coefficients (dirSubgrad entries)
    subgradientPath.inactives(1:size(find(setActive == 0)), 1) = ...
        find(setActive == 0);
    % calculate rho*subgrad
    subgradientPath.rhoSubgrad(:, 1) = rhopath(1)*subgrad;

    % set initial violations counter to 0
    violationsPath(1) = 0;
    
    % sign in path direction
    % dirsgn = 1;
    dirsgn = -1;
    % initialize k for manually looking at path following loop
    k = 2;
end

% main loop for path following
s = warning('error', 'MATLAB:nearlySingularMatrix'); %#ok<CTPCT>
for k = 2:maxiters 
%    tic; % timing for speed issues


    if strcmpi(direction, 'decrease')
        % threshold near-zero rhos to zero and stop algorithm
        if rhopath(k-1) <= (0 + 1e-8)
            rhopath(k-1) = 0;
            break;
        end
    end
    
    %# Calculate derivative for coefficients and multipliers (Eq. 8) #%
    % construct matrix
    M = [H(setActive, setActive) Aeq(:,setActive)' ...
        A(setIneqBorder,setActive)']; 
    M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ... 
        [Aeq(:,setActive); A(setIneqBorder,setActive)];
    % calculate derivative 
    try
%         % original code (regular inverse of M):
%         dir = dirsgn ...
%               * (M \ [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
       
%         % second code (pinv of M):
%         dir = dirsgn ...
%             * (pinv(M) * ...
%             [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
             
        % third code (derivative sign defined in terms of rho increasing)
        dir = -(pinv(M) * ...
            [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%         % make sure values from both methods match
%         if sum(abs(dir) ~= abs(dir)) ~= 0
%             warning('dir values dont match')
%             display(k)
%             break
%        end
    catch
        break;
    end
%     % possible fix for numerical issues (thresholding small derivatives to
%     % zero.  This did not work well.  
%     dir(abs(dir) < 1e-12) = 0;
       
    %# calculate derivative for rho*subgradient (Eq. 10) #%
%     % original code
%     dirSubgrad = ...
%          - [H(~setActive, setActive) Aeq(:,~setActive)' ...
%          A(setIneqBorder,~setActive)'] * dir;
    % second code (derivative sign defined in terms of rho increasing)
    dirSubgrad = ...
        - [H(~setActive, setActive) Aeq(:,~setActive)' ...
        A(setIneqBorder,~setActive)'] * dir;
%     % make sure values from both methods match
%     if sum(abs(dirSubgrad) ~= abs(dirSubgrad)) ~= 0
%         warning('dirSubgrad values dont match')
%         display(k)
%         break
%     end
    
    
    %## check to see if any conditions are violated ##%

    %# Inactive coefficients moving too slowly #%
%     % Negative subgradient (original code)
%     inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
%         subgrad(~setActive) <= (-1 + 1e-8) & 0 < dirSubgrad & ...
%         dirSubgrad < 1);
    % Negative subgradient (new code - original bounds)
    inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad & ...
        dirSubgrad < 0);
    % Negative subgradient (new code - new bounds)
    inactSlowNegIdx2 = find((-1 - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad);
    % make sure values from both methods match
    if sum(inactSlowNegIdx ~= inactSlowNegIdx2) ~= 0
        warning('inactSlowNegIdx values dont match')
        display(k)
        break
    end
    %%%%%%%%%%%%%%%%%% Possibly change ALL bounds to account for numerical
    %%%%%%%%%%%%%%%%%% error
    
%     % Positive subgradient
%     inactSlowPosIdx = find((1 - 1e-8) <= subgrad(~setActive) & ...
%         subgrad(~setActive) <= (1 + 1e-8) & -1 < dirSubgrad & ...
%         dirSubgrad < 0);
    % Positive subgradient (new code - original bounds)
    inactSlowPosIdx = find((1 - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (1 + 1e-8) & 0 < dirSubgrad & ...
        dirSubgrad < 1);
    % Positive subgradient (new code - new bounds)
    inactSlowPosIdx2 = find((1 - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (1 + 1e-8) & dirSubgrad < 1);
    % make sure values from both methods match
    if sum(inactSlowPosIdx ~= inactSlowPosIdx2) ~= 0
        warning('inactSlowPosIdx values dont match')
        display(k)
        break
    end
    
    %# "Active" coeficients estimated as 0 with potential sign mismatch #%
%     % Positive subgrad but negative derivative 
%     signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
%         subgrad(setActive) <= (1 + 1e-8) & dir(1:nActive) <= (0 - 1e-8) & ...
%         betapath(setActive, k-1) == 0);   
    % Positive subgrad but positive derivative (new code)
    signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
        subgrad(setActive) <= (1 + 1e-8) & (0 + 1e-8) <= dir(1:nActive) & ...
        betapath(setActive, k-1) == 0);
%     % make sure values from both methods match
%     if sum(signMismatchPosIdx ~= signMismatchPosIdx) ~= 0
%         warning('signMismatchPosIdx values dont match')
%         display(k)
%         break
%     end    
    
%     % Negative subgradient but positive derivative
%     signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
%         subgrad(setActive) <= (0 + 1e-8) & (0 + 1e-8) <=  dir(1:nActive) & ...
%         betapath(setActive, k-1) == 0);
    % Negative subgradient but negative derivative (new code)
    signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
        subgrad(setActive) <= (0 + 1e-8) & dir(1:nActive) <= (0 - 1e-8) & ...
        betapath(setActive, k-1) == 0);
%     % make sure values from both methods match
%     if sum(signMismatchNegIdx ~= signMismatchNegIdx) ~= 0
%         warning('signMismatchNegIdx values dont match')
%         display(k)
%         break
%     end
    
%     %# temp stuff for debugging #%
%     % specify trouble coefficient
%     troubleCoeff = 9;
%     % check active status of coefficient 
%     setActive(troubleCoeff);
%     
%     % value of subgradient
%     subgrad(troubleCoeff); % I don't understand this error
%     % betahat_{troubleCoeff}
%     betapath(troubleCoeff, k-1);
%     
%     % define which coefficients are inactive 
%     inactives = find(setActive == 0);
%     % define which coefficients are inactive 
%     actives = find(setActive == 1);
%     
%     % find the index of the inactive set corresponding to the trouble
%     % coefficient (if applicable) 
%     troubleIdxInactive = find(inactives == troubleCoeff);
%     % derivative of rho*subgrad for trouble coefficient 
%     dirSubgrad(troubleIdxInactive); % I don't understand this error
% 
%     % find the index of the active set corresponding to the trouble
%     % coefficient (if applicable)    
%     troubleIdxActive = find(actives == troubleCoeff);
%     % derivative 
%     dir(troubleIdxActive);
            
    % reset violation counter (to avoid infinite loops)
    violateCounter = 0;
    
    %# Outer while loop for checking all conditions together #%
    while ~isempty(inactSlowNegIdx) || ~isempty(inactSlowPosIdx) || ...
            ~isempty(signMismatchPosIdx) || ~isempty(signMismatchNegIdx)
    
        % Monitor & fix condition 1 violations
        while ~isempty(inactSlowNegIdx)
         
            %# Identify & move problem coefficient #%
            % indices corresponding to inactive coefficients
            inactiveCoeffs = find(setActive == 0);
            % identify prblem coefficient
            viol_coeff = inactiveCoeffs(inactSlowNegIdx);
            % put problem coefficient back into active set;
            setActive(viol_coeff) = true;
            % determine new number of active coefficients
            nActive = nnz(setActive);
            
            %# Recalculate derivative for coefficients & multipliers (Eq. 8) #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
%                 % original code (regular inverse of M):
%                 dir = dirsgn ...
%                     * (M \ ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
                
%                 % second code (pinv of M):
%                 dir = dirsgn ...
%                     * (pinv(M) * ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
                
                % 3rd code (derivative sign defined in terms of rho increasing)
                dir = -(pinv(M) * ...
                    [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 % make sure values from both methods match
%                 if sum(abs(dir) ~= abs(dir)) ~= 0
%                     warning('dir values dont match')
%                     display(k)
%                     break
%                 end
            catch
                break;
            end
%             % possible fix for numerical issues (thresholding small 
%             % derivatives to zero.  This did not work well.
%             dir(abs(dir) < 1e-12) = 0;
            
            %# calculate derivative for rho*subgradient (Eq. 10) #%
%             % original code
%             dirSubgrad = ...
%                 - [H(~setActive, setActive) Aeq(:,~setActive)' ...
%                 A(setIneqBorder,~setActive)'] * dir;
            % second code (derivative sign defined in terms of rho increasing)
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
%             % make sure values from both methods match
%             if sum(abs(dirSubgrad) ~= abs(dirSubgrad)) ~= 0
%                 warning('dirSubgrad values dont match')
%                 display(k)
%                 break
%             end
            
            %# Misc. housekeeping #%
%             % check for violations again
%             inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
%                 subgrad(~setActive) <= (-1 + 1e-8) & ...
%                 0 < dirSubgrad & dirSubgrad < 1);
            % Negative subgradient (new code)
            inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad & ...
                dirSubgrad < 0);
            % Negative subgradient (new code - new bounds)
            inactSlowNegIdx2 = find((-1 - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad);
            % make sure values from both methods match
            if sum(inactSlowNegIdx ~= inactSlowNegIdx2) ~= 0
                warning('inactSlowNegIdx values dont match')
                display(k)
                break
            end
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
        end

        % Monitor & fix condition 2 violations
        while ~isempty(inactSlowPosIdx)
            
            %# Identify & move problem coefficient #%
            % indices corresponding to inactive coefficients
            inactiveCoeffs = find(setActive == 0);
            % identify problem coefficient
            viol_coeff = inactiveCoeffs(inactSlowPosIdx);
            % put problem coefficient back into active set;
            setActive(viol_coeff) = true;
            % determine new number of active coefficients
            nActive = nnz(setActive);
                        
            %# Recalculate derivative for coefficients & multipliers (Eq. 8) #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
%                 % original code (regular inverse of M):
%                 dir = dirsgn ...
%                     * (M \ ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
                
%                 % second code (pinv of M):
%                 dir = dirsgn ...
%                     * (pinv(M) * ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
                
                % 3rd code (derivative sign defined in terms of rho increasing)
                dir = -(pinv(M) * ...
                    [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 % make sure values from both methods match
%                 if sum(abs(dir) ~= abs(dir)) ~= 0
%                     warning('dir values dont match')
%                     display(k)
%                     break
%                 end
            catch
                break;
            end
%             % possible fix for numerical issues (thresholding small 
%             % derivatives to zero.  This did not work well.
%             dir(abs(dir) < 1e-12) = 0;
            
            %# calculate derivative for rho*subgradient (Eq. 10) #%
%             % original code
%             dirSubgrad = ...
%                 - [H(~setActive, setActive) Aeq(:,~setActive)' ...
%                 A(setIneqBorder,~setActive)'] * dir;
            % second code (derivative sign defined in terms of rho increasing)
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
%             % make sure values from both methods match
%             if sum(abs(dirSubgrad) ~= abs(dirSubgrad)) ~= 0
%                 warning('dirSubgrad values dont match')
%                 display(k)
%                 break
%             end
            
            %# Misc. housekeeping #%
%             % check for violations again
%             inactSlowPosIdx = find(subgrad(~setActive) == 1 & ...
%                 -1 < dirSubgrad & dirSubgrad < 0);
            % Positive subgradient (new code)
            inactSlowPosIdx = find((1 - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (1 + 1e-8) & 0 < dirSubgrad & ...
                dirSubgrad < 1);
            % Positive subgradient (new code - new bounds)
            inactSlowPosIdx2 = find((1 - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (1 + 1e-8) & dirSubgrad < 1);
            % make sure values from both methods match
            if sum(inactSlowPosIdx ~= inactSlowPosIdx2) ~= 0
                warning('inactSlowPosIdx values dont match')
                display(k)
                break
            end
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
        end
        
        % Monitor & fix condition 3 violations
        while ~isempty(signMismatchPosIdx)
            
            %# Identify & move problem coefficient #%
            % indices corresponding to active coefficients
            activeCoeffs = find(setActive == 1);
            % identify prblem coefficient
            viol_coeff = activeCoeffs(signMismatchPosIdx);
            % put problem coefficient back into inactive set;
            setActive(viol_coeff) = false;
            % determine new number of active coefficients
            nActive = nnz(setActive);
            
            
            %# Recalculate derivative for coefficients & multipliers (Eq. 8) #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
%                 % original code (regular inverse of M):
%                 dir = dirsgn ...
%                     * (M \ ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 
%                 % second code (pinv of M):
%                 dir = dirsgn ...
%                     * (pinv(M) * ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 
                % 3rd code (derivative sign defined in terms of rho increasing)
                dir = -(pinv(M) * ...
                    [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 % make sure values from both methods match
%                 if sum(abs(dir) ~= abs(dir)) ~= 0
%                     warning('dir values dont match')
%                     display(k)
%                     break
%                 end
            catch
                break;
            end
%             % possible fix for numerical issues (thresholding small 
%             % derivatives to zero.  This did not work well.
%             dir(abs(dir) < 1e-12) = 0;
            
            %# calculate derivative for rho*subgradient (Eq. 10) #%
%             % original code
%             dirSubgrad = ...
%                 - [H(~setActive, setActive) Aeq(:,~setActive)' ...
%                 A(setIneqBorder,~setActive)'] * dir;
            % second code (derivative sign defined in terms of rho increasing)
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
%             % make sure values from both methods match
%             if sum(abs(dirSubgrad) ~= abs(dirSubgrad)) ~= 0
%                 warning('dirSubgrad values dont match')
%                 display(k)
%                 break
%             end
            
            %# Misc. housekeeping #%
            % check for violations again
%             signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
%                 subgrad(setActive) <= (1 + 1e-8) & ...
%                 dir(1:nActive) <= (0 - 1e-8) & ...
%                 betapath(setActive, k-1) == 0);
            % Positive subgrad but positive derivative (new code)
            signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (1 + 1e-8) & (0 + 1e-8) <= dir(1:nActive) & ...
                betapath(setActive, k-1) == 0);
%             % make sure values from both methods match
%             if sum(signMismatchPosIdx ~= signMismatchPosIdx) ~= 0
%                 warning('signMismatchPosIdx values dont match')
%                 display(k)
%                 break
%             end
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
        end
        
        % Monitor & fix condition 4 violations
        while ~isempty(signMismatchNegIdx)
            
            %# Identify & move problem coefficient #%
            % indices corresponding to active coefficients
            activeCoeffs = find(setActive == 1);
            % identify prblem coefficient
            viol_coeff = activeCoeffs(signMismatchNegIdx);
            % put problem coefficient back into inactive set;
            setActive(viol_coeff) = false;
            % determine new number of active coefficients
            nActive = nnz(setActive);
            
            
            %# Recalculate derivative for coefficients & multipliers (Eq. 8) #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nnz(setIneqBorder), 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
%                 % original code (regular inverse of M):
%                 dir = dirsgn ...
%                     * (M \ ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
                
%                 % second code (pinv of M):
%                 dir = dirsgn ...
%                     * (pinv(M) * ...
%                     [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 
                % 3rd code (derivative sign defined in terms of rho increasing)
                dir = -(pinv(M) * ...
                    [subgrad(setActive); zeros(m1+nnz(setIneqBorder),1)]);
%                 % make sure values from both methods match
%                 if sum(abs(dir) ~= abs(dir)) ~= 0
%                     warning('dir values dont match')
%                     display(k)
%                     break
%                 end
            catch
                break;
            end
%             % possible fix for numerical issues (thresholding small 
%             % derivatives to zero.  This did not work well.
%             dir(abs(dir) < 1e-12) = 0;
            
            %# calculate derivative for rho*subgradient (Eq. 10) #%
%             % original code
%             dirSubgrad = ...
%                 - [H(~setActive, setActive) Aeq(:,~setActive)' ...
%                 A(setIneqBorder,~setActive)'] * dir;
            % second code (derivative sign defined in terms of rho increasing)
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
%                     % make sure values from both methods match
%             if sum(abs(dirSubgrad) ~= abs(dirSubgrad)) ~= 0
%                 warning('dirSubgrad values dont match')
%                 display(k)
%                 break
%             end
            
            %# Misc. housekeeping #%
%             % check for violations again
%             signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
%                 subgrad(setActive) <= (0 + 1e-8) & ...
%                 (0 + 1e-8) <=  dir(1:nActive) & ...
%                 betapath(setActive, k-1) == 0, 1);
            % Negative subgradient but negative derivative (new code)
            signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (0 + 1e-8) & ...
                dir(1:nActive) <= (0 - 1e-8) & betapath(setActive, k-1) == 0);
%             % make sure values from both methods match
%             if sum(signMismatchNegIdx ~= signMismatchNegIdx) ~= 0
%                 warning('signMismatchNegIdx values dont match')
%                 display(k)
%                 break
%             end
            
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
            
        end
        
        
        %## update violation trackers to see if any issues persist ##%
        
        %# Inactive coefficients moving too slowly #%
%         % Negative subgradient
%          inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
%             subgrad(~setActive) <= (-1 + 1e-8) & 0 < dirSubgrad & ...
%             dirSubgrad < 1);
        % Negative subgradient (new code)
        inactSlowNegIdx = find((-1 - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad & ...
            dirSubgrad < 0);
        % Negative subgradient (new code - new bounds)
        inactSlowNegIdx2 = find((-1 - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (-1 + 1e-8) & -1 < dirSubgrad);
        % make sure values from both methods match
        if sum(inactSlowNegIdx ~= inactSlowNegIdx2) ~= 0
            warning('inactSlowNegIdx values dont match')
            display(k)
            break
        end

%         % Positive subgradient
%         inactSlowPosIdx = find((1 - 1e-8) <= subgrad(~setActive) & ...
%             subgrad(~setActive) <= (1 + 1e-8) & -1 < dirSubgrad & ...
%             dirSubgrad < 0);
        % Positive subgradient (new code)
        inactSlowPosIdx = find((1 - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (1 + 1e-8) & 0 < dirSubgrad & ...
            dirSubgrad < 1);
        % Positive subgradient (new code - new bounds)
        inactSlowPosIdx2 = find((1 - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (1 + 1e-8) & dirSubgrad < 1);
        % make sure values from both methods match
        if sum(inactSlowPosIdx ~= inactSlowPosIdx2) ~= 0
            warning('inactSlowPosIdx values dont match')
            display(k)
            break
        end
            
        %# "Active" coeficients estimated as 0 with potential sign mismatch #%
%         % Positive subgrad but negative derivative
%         signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
%             subgrad(setActive) <= (1 + 1e-8) & ...
%             dir(1:nActive) <= (0 - 1e-8) & betapath(setActive, k-1) == 0);
        % Positive subgrad but positive derivative (new code)
        signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
            subgrad(setActive) <= (1 + 1e-8) & (0 + 1e-8) <= dir(1:nActive) & ...
            betapath(setActive, k-1) == 0);
%         % make sure values from both methods match
%         if sum(signMismatchPosIdx ~= signMismatchPosIdx) ~= 0
%             warning('signMismatchPosIdx values dont match')
%             display(k)
%             break
%         end
%         % Negative subgradient but positive derivative
%         signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
%             subgrad(setActive) <= (0 + 1e-8) & ...
%             (0 + 1e-8) <=  dir(1:nActive) & betapath(setActive, k-1) == 0, 1);
        % Negative subgradient but negative derivative (new code)
        signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
            subgrad(setActive) <= (0 + 1e-8) & dir(1:nActive) <= (0 - 1e-8) & ...
            betapath(setActive, k-1) == 0);
%         % make sure values from both methods match
%         if sum(signMismatchNegIdx ~= signMismatchNegIdx) ~= 0
%             warning('signMismatchNegIdx values dont match')
%             display(k)
%             break
%         end
        
        % break loop if needed
        if violateCounter >= maxiters
            break;
        end
    end % end of outer while loop
    
    % store number of violations (for debugging(
    violationsPath(k) = violateCounter;
    
    
    % calculate derivative for residual inequality (Eq. 11)
%     % original code
%     dirResidIneq = A(~setIneqBorder, setActive)*dir(1:nActive);
    % new code
    dirResidIneq = A(~setIneqBorder, setActive)*dir(1:nActive);
 
    %### Determine rho for next event (via delta rho) ###%
    %## Events based on coefficients changing activation status ##%
%     
%     % clear previous values for delta rho
%     nextrhoBeta = inf(p, 1);
    nextrhoBeta = inf(p, 1);
    
    %# Active coefficient going inactive #%
%     % original code
%     nextrhoBeta(setActive) = -betapath(setActive, k-1) ...
%         ./ dir(1:nActive);
    % new code
    nextrhoBeta(setActive) = -dirsgn*betapath(setActive, k-1) ...
        ./ dir(1:nActive); 
%     % make sure values from both methods match
%     if sum(nextrhoBeta ~= nextrhoBeta) ~= 0
%         warning('delta rho values for beta dont match')
%         display(k)
%         break
%     end
    
    %# Inactive coefficient becoming positive #%
%     % original code
%     t1 = rhopath(k-1)*(1 - subgrad(~setActive)) ./ (dirSubgrad + dirsgn);
    % new code
    t1 = dirsgn*rhopath(k-1)*(1 - subgrad(~setActive)) ./ (dirSubgrad - 1);
%     % make sure values from both methods match
%     if sum(t1(~isinf(t1)) ~= t1New(~isinf(t1New))) ~= 0
%         warning('t1 values dont match')
%         display(k)
%         break
%     end
    % t1(5)==t1New(5)
    % threshold values hitting ceiling
%     t1(t1 <= (0 + 1e-8)) = inf;
    t1(t1 <= (0 + 1e-8)) = inf;

    %# Inactive coefficient becoming negative #%
%     % original code
%     t2 = rhopath(k-1)*(- 1 - subgrad(~setActive)) ...
%         ./ (dirSubgrad - dirsgn);
    % new code
    t2 = -dirsgn*rhopath(k-1)*(1 + subgrad(~setActive)) ...
        ./ (dirSubgrad + 1);
%     % make sure values from both methods match
%     if sum(t2 ~= t2) ~= 0
%         warning('t2 values dont match')
%         display(k)
%         break
%     end
    % threshold values hitting ceiling
%     t2(t2 <= (0 + 1e-8)) = inf;
    t2(t2 <= (0 + 1e-8)) = inf;        
        
    % choose smaller delta rho out of t1 and t2
%     nextrhoBeta(~setActive) = min(t1, t2);
    nextrhoBeta(~setActive) = min(t1, t2);
    % nextrhoBeta(troubleIdxInactive);
%     % make sure values from both methods match
%     if sum(nextrhoBeta ~= nextrhoBeta) ~= 0
%         warning('nextrhoBeta values dont match')
%         display(k)
%         break
%     end
    
    % ignore delta rhos equal to zero
%     nextrhoBeta(nextrhoBeta <= 1e-8 | ~penidx) = inf;
    nextrhoBeta(nextrhoBeta <= 1e-8 | ~penidx) = inf;
%     % make sure values from both methods match
%     if sum(nextrhoBeta ~= nextrhoBeta) ~= 0
%         warning('nextrhoBeta values dont match')
%         display(k)
%         break
%     end
    
    %## Events based inequality constraints ##%
	% clear previous values
%     nextrhoIneq = inf(m2, 1);
    nextrhoIneq = inf(m2, 1);
    
    %# Inactive inequality constraint becoming active #%
%     % original code
%     nextrhoIneq(~setIneqBorder) = - residIneq(~setIneqBorder) ...
%         ./ dirResidIneq;
    % new code 
    nextrhoIneq(~setIneqBorder) = -dirsgn*residIneq(~setIneqBorder) ...
        ./ dirResidIneq;    
    
    %# Active inequality constraint becoming deactive #%
%     % original code   
%     nextrhoIneq(setIneqBorder) = - dualpathIneq(setIneqBorder, k-1) ...
%         ./ reshape(dir(nActive+m1+1:end), nnz(setIneqBorder),1);     
    % new code
    nextrhoIneq(setIneqBorder) = - dualpathIneq(setIneqBorder, k-1) ...
        ./ reshape(dir(nActive+m1+1:end), nnz(setIneqBorder),1);     
%     % make sure values from both methods match
%     if sum(nextrhoIneq ~= nextrhoIneq) ~= 0
%         warning('Inequality delta rho values dont match')
%         display(k)
%         break
%     end   
    
    % ignore delta rhos equal to zero
%     nextrhoIneq(nextrhoIneq <= 1e-8) = inf;
    nextrhoIneq(nextrhoIneq <= 1e-8) = inf;
%     
%     % make sure values from both methods match
%     if sum(nextrhoIneq ~= nextrhoIneq) ~= 0
%         warning('Inequality delta rho values dont match')
%         display(k)
%         break
%     end   
    
    %# determine next rho ##
    % original method:
    if strcmpi(multCoeff, 'false')
        [chgrho, idx] = min([nextrhoBeta; nextrhoIneq]);
    
    % new method picking multiple rhos:
    elseif strcmpi(multCoeff, 'true')
        % find smallest rho
        chgrho = min([nextrhoBeta; nextrhoIneq]);
        % find all indices corresponding to this chgrho
        idx = find(([nextrhoBeta; nextrhoIneq] - chgrho) <= deltaRhoTol);
    end
    
    % terminate path following if no new event found
    if isinf(chgrho)
        break;
    end
    
    
    %## Update values at new rho ##%
    %# move to next rho #%
    % make sure next rho isn't negative
%     % original code
%     if rhopath(k-1) - dirsgn*chgrho < 0 % may wanna change to maxrho
%         chgrho = rhopath(k-1);              % for increasing direction?
%     end
%     rhopath(k) = rhopath(k-1) - dirsgn*chgrho;
    % new code
    if rhopath(k-1) + dirsgn*chgrho < 0 % may wanna change to maxrho
        chgrho = rhopath(k-1);              % for increasing direction?
    end
%     % make sure values from both methods match
%     if sum(rhopath(k) ~= rhopath(k-1) + dirsgn*chgrho) ~= 0
%         warning('nextrhoBeta values dont match')
%         display(k)
%         break
%     end
    rhopath(k) = rhopath(k-1) + dirsgn*chgrho;
    
    %# Update parameter and subgradient values #%
    % new coefficient estimates
%     % original code
%     betapath(setActive, k) = betapath(setActive, k-1) ...
%          + chgrho*dir(1:nActive);
%     % make sure values from both methods match
%     if sum(betapath(setActive, k) ~= (betapath(setActive, k-1) ...
%        + dirsgn*chgrho*dir(1:nActive))) ~= 0
%         warning('beta estimates dont match')
%         display(k)
%         break
%     end    
    % new code
    betapath(setActive, k) = betapath(setActive, k-1) ...
        + dirsgn*chgrho*dir(1:nActive);
    % force near-zero coefficients to be zero (helps with numerical issues)
    betapath(abs(betapath(:, k)) < 1e-12, k) = 0; 
    
    % new subgradient estimates
%     % create subgrad vector for new code
%     subgrad2 = subgrad;
%     % original code
%     subgrad(~setActive) = ...
%         (rhopath(k-1)*subgrad(~setActive) + chgrho*dirSubgrad)/rhopath(k);   
    % new code
    subgrad(~setActive) = ...
        (rhopath(k-1)*subgrad(~setActive) + dirsgn*chgrho*dirSubgrad)/rhopath(k);
%     % make sure new code matches old
%     if sum(subgrad(~setActive) ~= subgrad(~setActive)) ~= 0
%         warning('subgradients dont match')
%         display(k)
%         break
%     end     
 
    %###### left off here ############%
    % update lambda (lagrange multipliers for equality constraints)
%     % original code
%     dualpathEq(:, k) = dualpathEq(:,k-1) ...
%         + chgrho*reshape(dir(nActive+1:nActive+m1),m1,1);
    % new code
    dualpathEq(:, k) = dualpathEq(:, k-1) ...
        + dirsgn*chgrho*reshape(dir(nActive+1:nActive+m1), m1, 1);
%     % make sure new code matches old
%     if sum(dualpathEq(:, k) ~= dualpathEq(:, k)) ~= 0
%         warning('lambda values dont match')
%         display(k)
%         break
%     end   
    
    % update mu (lagrange multipliers for inequality constraints)
%     % original code
%     dualpathIneq(setIneqBorder, k) = dualpathIneq(setIneqBorder, k-1) ...
%         + chgrho*reshape(dir(nActive+m1+1:end), nnz(setIneqBorder),1);
    % new code
    dualpathIneq(setIneqBorder, k) = dualpathIneq(setIneqBorder, k-1) ...
        + dirsgn*chgrho*reshape(dir(nActive+m1+1:end), nnz(setIneqBorder), 1);   
%     % make sure new code matches old
%     if sum(dualpathIneq(setIneqBorder, k) ~= ...
%         dualpathIneq(setIneqBorder, k)) ~= 0
%             warning('lambda values dont match')
%             display(k)
%             break
%     end   
    
    % update residual inequality
    residIneq = A*betapath(:, k) - b;
    
    
%     %## Stuff for Debugging ##%
%     % specify trouble coefficient
%     troubleCoeff = 9;
%     % check active status of coefficient 
%     setActive(troubleCoeff);
%     
%     % value of subgradient
%     subgrad(troubleCoeff); % I don't understand this error
%     % betahat_{troubleCoeff}
%     betapath(troubleCoeff, k);
%     
%     % define which coefficients are inactive 
%     inactives = find(setActive == 0);
%     % define which coefficients are inactive 
%     actives = find(setActive == 1);
%     
%     % find the index of the inactive set corresponding to the trouble
%     % coefficient (if applicable) 
%     troubleIdxInactive = find(inactives == troubleCoeff);
%     % derivative of rho*subgrad for trouble coefficient 
%     dirSubgrad(troubleIdxInactive); % I don't understand this error
% 
%     % find the index of the active set corresponding to the trouble
%     % coefficient (if applicable)    
%     troubleIdxActive = find(actives == troubleCoeff);
%     % derivative 
%     dir(troubleIdxActive);
 
    
    %## update sets ##%
    if strcmpi(multCoeff, 'false')
        %# old code: #%
        if idx <= p && setActive(idx)
            % an active coefficient hits 0, or
            setActive(idx) = false;
        elseif idx <= p && ~setActive(idx)
            % a zero coefficient becomes nonzero
            setActive(idx) = true;
        elseif idx > p
            % an ineq on boundary becomes strict, or
            % a strict ineq hits boundary
            setIneqBorder(idx-p) = ~setIneqBorder(idx-p);
        end
        
    elseif strcmpi(multCoeff, 'true')
        %# new code(to try to allow for multiple coefficients moving #%
        for j = 1:length(idx)
            curidx = idx(j);
            if curidx <= p && setActive(curidx)
                % an active coefficient hits 0, or
                setActive(curidx) = false;
            elseif curidx <= p && ~setActive(curidx)
                % a zero coefficient becomes nonzero
                setActive(curidx) = true;
            elseif curidx > p
                % an ineq on boundary becomes strict, or
                % a strict ineq hits boundary
                setIneqBorder(curidx-p) = ~setIneqBorder(curidx-p);
            end
        end
    end
    
    % determine new number of active coefficients
    nActive = nnz(setActive);
            
    % old code
%     setActive = abs(betapath(:,k))>1e-16 | ~penidx;
%     betapath(~setActive,k) = 0;

    % calculate value of objective function
    objValPath(k) = norm(y - X*betapath(:, k))^2/2 + ...
        rhopath(k)*sum(abs(betapath(:, k)));
     
    % calculate degrees of freedom (using two different methods, I believe
    % method 1 is more correct).  Also, df are thresholded at zero.  
    dfPath(1, k) = rank(X(:,  setActive)) - rankAeq;
    dfPath(2, k) = nActive - rankAeq;
    %dfPath(1, k) = max(rank(X(:,  setActive)) - rankAeq, 0);
    %dfPath(2, k) = max(nActive - rankAeq, 0);
    % break algorithm when df are exhausted
    if dfPath(2, k) > n
        break;
    end
    

    %## Calculate & store stuff for debuggin ##%
    %# Stationarity conditions #%
    % calculate the stationarity condition value
    stationarityConditionsPath.values(:, k) = -X'*(y - X*betapath(:, k)) + ...
        rhopath(k)*subgrad + Aeq'*dualpathEq(:, k) + A'*dualpathIneq(:, k);
    % see if stationarity condition is satisified
    stationarityConditionsPath.satisfied(k) = ...
        sum(abs(stationarityConditionsPath.values(:, k)) < 1e-8) == p;
    
    %# Constraints #%
    % check if constraints are violated or not
    % equality
    if m1==0
        constraintsSatisfied.eq(k) = NaN;
    else
        constraintsSatisfied.eq(k) = ...
            sum(abs(Aeq*betapath(:, k) - beq) < 1e-10) == m1;
    end
    % inequality
    if m2==0
        constraintsSatisfied.ineq(k) = NaN;
    else
        constraintsSatisfied.ineq(k) = ...
            sum(A*betapath(:, k) - b < 1e-10) == m2;
    end
    
    %# Subgradient #%
    % store subgradient
    subgradientPath.values(:, k) = subgrad;
    % check that subgradient condition is satisfied (both in [-1, 1] and
        % matching the sign
    subgradientPath.satisfied(k) = ...
        nnz(subgradientPath.values(:, k) <= (1 + 1e-8) & ...
        subgradientPath.values(:, k) >= (-1 - 1e-8) & ... 
        betapath(:, k).*subgrad >= 0) == p;
    % derivative for subgradient
    % uses k-1 since this is calculated at the beginning of the loop
    subgradientPath.dir(1:size(dirSubgrad, 1), k-1) = dirSubgrad;
    % indices for inactive coefficients (dirSubgrad entries)
    % uses k since setActive has already been updated
    subgradientPath.inactives(1:size(find(setActive == 0)), k) = ...
        find(setActive == 0);
    % calculate & store rho*subgrad  
    subgradientPath.rhoSubgrad(:, k) = rhopath(k)*subgrad;
    
   
    % # old debugging code #%
    % manually check that the subgradient sign matches the coefficients
    %find(abs(subgrad(setActive) - sign(betapath(setActive,k))) > 1e-12)
    %       find(abs(subgrad(setActive) - sign(betapath(setActive,k))) > 1e-12 & ...
       % sign(betapath(setActive,k)) ~= 0);
%     idxSubgradWrong = ...
%         find(abs(subgrad - sign(betapath(:,k))) > 1e-12 & ...
%         sign(betapath(:,k)) ~= 0);
%     subgrad(idxSubgradWrong) = -subgrad(idxSubgradWrong);
%     
%     
   
    %toc
end

% clean up
warning(s);
betapath(:, k:end) = [];
dualpathEq(:, k:end) = [];
dualpathIneq(:, k:end) = [];
rhopath(k:end) = [];
objValPath(k:end) = [];
stationarityConditionsPath.values(:, k:end) = [];
stationarityConditionsPath.satisfied(k:end) = [];
dfPath(:, k:end) = [];
constraintsSatisfied.eq(k:end) = [];
constraintsSatisfied.ineq(k:end) = [];
subgradientPath.values(:, k:end) = [];
subgradientPath.satisfied(k:end) = [];
subgradientPath.dir(:, k:end) = [];
subgradientPath.inactives(:, k:end) = [];
subgradientPath.rhoSubgrad(:, k:end) = [];
violationsPath(k:end) = [];
    
    
end