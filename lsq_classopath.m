function [rhoPath, betaPath, dfPath, objValPath] = ...
    lsq_classopath(X, y, A, b, Aeq, beq, varargin)
% LSQ_CLASSOPATH Constrained lasso solution path
%
%   [RHO_PATH, BETA_PATH] = LSQ_CLASSOPATH(X, y, A, b, Aeq, beq) computes the 
%   solution path for the constrained lasso problem using the predictor
%   matrix X and response y.  The constrained lasso solves the standard
%   lasso (Tibshirani, 1996) subject to the linear equality constraints
%   Aeq*beta = beq and linear inequality constraints A*beta <= b.  The
%   result RHO_PATH contains the values of the tuning parameter rho along
%   the solution path.  The result BETA_PATH has a vector of the estimated
%   regression coefficients for each value of rho.
%
%   [RHO_PATH, BETA_PATH] = LSQ_CLASSOPATH(X, y, A, b, Aeq, beq, ...
%   'PARAM1', val1, 'PARAM2', val2, ...) allows you to specify optional
%   parameter name/value pairs to control the model fit.
%
%
% INPUTS:
%   X: n-by-p design matrix
%   y: n-by-1 response vector
%   A: inequality constraint matrix
%   b: inequality constraint vector
%   Aeq: equality constraint matrix
%   beq: equality constraint vector
%
% OPTIONAL NAME-VALUE PAIRS:      
%
%   'qp_solver': 'matlab' (default), 'gurobi' currently is not working
%   'epsilon': tuning parameter for ridge penalty.  Default is 1e-4.
%
% OUTPUTS:
%
%   rhoPath: vector of the tuning parameter values along the solution path
%   betaPath: matrix with estimated regression coefficients for each value
%       of rho
%   dfPath: vector with estimate of degrees of freedom along the solution path
%   objValPath: vector of values of the objective function at each value of rho 
%       along the solution path
%
% EXAMPLE
%   See tutorial examples at https://github.com/Hua-Zhou/SparseReg
%
% REFERENCES
%   Gaines, Brian and Zhou, Hua (2016). On Fitting the Constrained Lasso.  
%
%
% Copyright 2014-2017 University of California at Los Angeles and North Carolina State University
% Hua Zhou (huazhou@ucla.edu) and Brian Gaines (brgaines@ncsu.edu)
%

% input parsing rule
[n, p] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('A', @(x) size(x,2)==p || isempty(x));
argin.addRequired('b', @(x) isnumeric(x) || isempty(x));
argin.addRequired('Aeq', @(x) size(x,2)==p || isempty(x));
argin.addRequired('beq', @(x) isnumeric(x) || isempty(x));
argin.addParamValue('qp_solver', 'matlab', @ischar);
argin.addParamValue('epsilon', 1e-4, @isnumeric);

% parse inputs
y = reshape(y, n, 1);
argin.parse(X, y, A, b, Aeq, beq, varargin{:});
qp_solver = argin.Results.qp_solver;
epsilon = argin.Results.epsilon;

% check validity of qp_solver
if ~(strcmpi(qp_solver, 'matlab'))% || strcmpi(qp_solver, 'GUROBI'))
    error('sparsereg:lsq_classopath:qp_solver', ...
        'qp_solver not recognized');
end

% save original number of observations (for when ridge penalty is used)
n_orig = n;

% see if ridge penalty needs to be included
if n < p
    warning('Adding a small ridge penalty (default is 1e-4) since n < p')
    if epsilon <= 0
        warning('epsilon must be positive, switching to default value (1e-4)')
        epsilon = 1e-4;
    end
    % create augmented data
    y = [y; zeros(p, 1)];
    X = [X; sqrt(epsilon)*eye(p)];
    % record original number of observations
    n_orig = n;
else
    % make sure X is full column rank
    [~, R] = qr(X, 0);
    rankX = sum(abs(diag(R)) > abs(R(1))*max(n, p)*eps(class(R)));
    if (rankX ~= p)
        warning(['Adding a small ridge penalty (default is 1e-4) since X ' ...
            'is rank deficient']);
        if epsilon <= 0
            warning(['epsilon must be positive, switching to default value' ...
                ' (1e-4)'])
            epsilon = 1e-4;
        end
        % create augmented data
        y = [y; zeros(p, 1)];
        X = [X; sqrt(epsilon)*eye(p)];
        % record original number of observations
        n_orig = n;
    end
end
    

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
betaPath = zeros(p, maxiters);
dualpathEq = zeros(m1, maxiters);
dualpathIneq = zeros(m2, maxiters);
rhoPath = zeros(1, maxiters);
dfPath = Inf(1, maxiters);
objValPath = zeros(1, maxiters);
violationsPath = Inf(1, maxiters);


%# intialization (LP based for all cases)
H = X'*X;

    % compute solution for rho->inf via LP
    [x,~,~,~,lambda] = ...
        linprog(ones(2*p,1), [A -A], b, [Aeq -Aeq], beq, ...
        zeros(2*p,1), inf(2*p,1));
    betaPath(:,1) = x(1:p) - x(p+1:end);

    % if all coefficients initialize at zero, use Rosset and Zhu (2007) type
    % initialization
    numZero = 1e-6; % set threshold for numerically equal to zero
    if sum(abs(betaPath(:,1))>numZero)==0 % 
        dualpathEq(:,1) = lambda.eqlin; %every real number works here
        dualpathIneq(:,1) = lambda.ineqlin; % lambda.ineqlin from LP ensures compl. slackness so this choice works
        resid = y - X*betaPath(:, 1);
        subgrad = X'*resid - Aeq'*dualpathEq(:,1) - A'*dualpathIneq(:,1);
        rhoPath(1) = max(abs(subgrad));
        
    else % if at least one coefficient initializes not at zero, use second LP to obtain 
         % valid Lagrange multipliers (LMs)

        % identify sets
        idtBiggerZero = betaPath(:,1) > numZero;
        idtSmallerZero = betaPath(:,1) < -numZero;
        idtNotActive = abs(betaPath(:,1)) <= numZero;
        idtActive = idtNotActive == 0;
        sActive = idtBiggerZero - idtSmallerZero;
        sActive(sActive==0)=[];
        nInactive = sum(idtNotActive);

        % identify inequalities that bind at rho->inf solution
        idtBinding = abs(A * betaPath(:,1) - b) < numZero;
        nBinding = sum(idtBinding);
        
        % set up LP over LMs 
        resid = y - X*betaPath(:, 1);
        r = - X'*resid;
        V = [Aeq' A(idtBinding,:)'];
        fInit = zeros(size(dualpathEq,1)+nBinding,1);
        Ainit = [V(idtActive,:);-V(idtActive,:); V(idtNotActive,:);-V(idtNotActive,:);zeros(nBinding,size(V(idtNotActive,:),2))];
        if nBinding>0
            Ainit(end-nBinding+1:end,end-nBinding+1:end) = -eye(nBinding); % Binding LMs for inequality constraints must be positive 
        end

        % set very large rhoStart and run LP. Repeat with larger rhoStart
        % if no solution could be found that solves the stationarity
        % condition
        rhoStart = 1000;
        statCondHolds = 0;
        while statCondHolds == 0

            % inside loop because it depends on rhoStart
            binit = [-r(idtActive,1)-rhoStart*sActive;+r(idtActive,1)+rhoStart*sActive;rhoStart * ones(nInactive,1) - r(idtNotActive,1); rhoStart * ones(nInactive,1) + r(idtNotActive,1); zeros(nBinding,1)];

            % run LP over LMs
            g = linprog(fInit,Ainit,binit);

            % remeber LMs
            dualpathEq(:,1) = g(1:size(Aeq,1));
            dualpathIneq(:,1) = zeros(size(A,1),1);
            dualpathIneq(idtBinding,1) = g(size(Aeq,1)+1:end);
            
            % get errors in stat. cond. equations 
            SSEStatCon = norm(r(idtActive,1) + rhoStart * sActive + V(idtActive,:)*g,2);

            %check whether stationarity condition of active coefficients
            %holds
            statCondHolds = abs(SSEStatCon) < numZero;
            if statCondHolds == 0
                rhoStart = rhoStart * 10;
            end    
        end

        rhoPath(1) = rhoStart;
        
    end
        
% initialize sets
dualpathIneq(dualpathIneq(:,1) < 0,1) = 0; % fix Gurobi negative dual variables
setActive = abs(betaPath(:,1)) > numZero;
betaPath(~setActive,1) = 0;
setIneqBorder = dualpathIneq(:,1) > numZero;
residIneq = A*betaPath(:,1) - b;
nIneqBorder = nnz(setIneqBorder);

%  initialize subgradient vector
resid = y - X*betaPath(:, 1);
subgrad = 1/rhoPath(1) * (X'*resid - Aeq'*dualpathEq(:,1) - A'*dualpathIneq(:,1));
nActive = nnz(setActive);

% calculate value for objective function
objValPath(1) = norm(y-X*betaPath(:,1))^2/2 + ...
    rhoPath(1)*sum(abs(betaPath(:,1)));
% calculate degrees of freedom 
rankAeq = rank(Aeq);                           % should do this more efficiently
dfPath(1) = nActive - rankAeq - nIneqBorder;

% set initial violations counter to 0
violationsPath(1) = 0;

% sign for path direction (originally went both ways, but increasing was
%   retired)
dirsgn = -1;


%####################################%
%### main loop for path following ###%
%####################################%
s = warning('error', 'MATLAB:nearlySingularMatrix'); %#ok<CTPCT>
s2 = warning('error', 'MATLAB:singularMatrix'); %#ok<CTPCT>

for k = 2:maxiters 

    % threshold near-zero rhos to zero and stop algorithm
    if rhoPath(k-1) <= (0 + 1e-3)
        rhoPath(k-1) = 0;
        break;
    end
    
    %# Calculate derivative for coefficients and multipliers #%
    % construct matrix
    M = [H(setActive, setActive) Aeq(:,setActive)' ...
        A(setIneqBorder,setActive)']; 
    M(end+1:end+m1+nIneqBorder, 1:nActive) = ... 
        [Aeq(:,setActive); A(setIneqBorder,setActive)];
    % calculate derivative 
    try
        % try using a regular inverse first
        dir = dirsgn ...
            * (M \ [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
    catch
        % otherwise use the moore-penrose inverse
        dir = -(IMqrginv(M) * ...
            [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
    end
       
    %# calculate derivative for rho*subgradient #%
    dirSubgrad = ...
        - [H(~setActive, setActive) Aeq(:,~setActive)' ...
        A(setIneqBorder,~setActive)'] * dir;
    
    
    %## check additional events related to potential subgradient violations ##%

    %# Inactive coefficients moving too slowly #%
    % Negative subgradient
    inactSlowNegIdx = find((1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (1*dirsgn + 1e-8) & 1*dirsgn < dirSubgrad);   
    
    % Positive subgradient
    inactSlowPosIdx = find((-1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
        subgrad(~setActive) <= (-1*dirsgn + 1e-8) & dirSubgrad < -1*dirsgn);
    
    %# "Active" coeficients estimated as 0 with potential sign mismatch #%
    % Positive subgrad but negative derivative 
    signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
        subgrad(setActive) <= (1 + 1e-8) & ...
        dirsgn*dir(1:nActive) <= (0 - 1e-8)  & ...
        betaPath(setActive, k-1) == 0);
    % Negative subgradient but positive derivative
    signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
        subgrad(setActive) <= (0 + 1e-8) & ...
        (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
        betaPath(setActive, k-1) == 0);
             
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
            % determine number of active/binding inequality constraints
            nIneqBorder = nnz(setIneqBorder);
            
            %# Recalculate derivative for coefficients & multipliers #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nIneqBorder, 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
                % try using a regular inverse first
                dir = dirsgn ...
                    * (M \ ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            catch
                % otherwise use moore-penrose inverse 
                dir = -(IMqrginv(M) * ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            end
            
            %# calculate derivative for rho*subgradient #%
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
            
            %# Misc. housekeeping #%
            % check for violations again

            %# Inactive coefficients moving too slowly #%
            % Negative subgradient
            inactSlowNegIdx = ...
                find((1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (1*dirsgn + 1e-8) & ...
                1*dirsgn < dirSubgrad);
            % Positive subgradient
            inactSlowPosIdx = ...
                find((-1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
                subgrad(~setActive) <= (-1*dirsgn + 1e-8) & ...
                dirSubgrad < -1*dirsgn);
            
            %# "Active" coeficients est'd as 0 with potential sign mismatch #%
            % Positive subgrad but negative derivative
            signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (1 + 1e-8) & ...
                dirsgn*dir(1:nActive) <= (0 - 1e-8)  & ...
                betaPath(setActive, k-1) == 0);
            % Negative subgradient but positive derivative
            signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (0 + 1e-8) & ...
                (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
                betaPath(setActive, k-1) == 0);
            
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
        end

        % Monitor & fix subgradient condition 2 violations
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
            % determine number of active/binding inequality constraints
            nIneqBorder = nnz(setIneqBorder);
            
            %# Recalculate derivative for coefficients & multiplier #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nIneqBorder, 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
                % try using a regular inverse first
                dir = dirsgn ...
                    * (M \ ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            catch
                dir = -(IMqrginv(M) * ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            end
            
            %# calculate derivative for rho*subgradient #%
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
            
            %# Misc. housekeeping #%
            % check for violations again
            inactSlowPosIdx = find((-1*dirsgn - 1e-8) <= ...
                subgrad(~setActive) & ...
                subgrad(~setActive) <= (-1*dirsgn + 1e-8) & ...
                dirSubgrad < -1*dirsgn);
            
            %# "Active" coeficients est'd as 0 with potential sign mismatch #%
            % Positive subgrad but negative derivative
            signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (1 + 1e-8) & ...
                dirsgn*dir(1:nActive) <= (0 - 1e-8)  & ...
                betaPath(setActive, k-1) == 0);
            % Negative subgradient but positive derivative
            signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (0 + 1e-8) & ...
                (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
                betaPath(setActive, k-1) == 0);
            
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
            % determine number of active/binding inequality constraints
            nIneqBorder = nnz(setIneqBorder);
            
            %# Recalculate derivative for coefficients & multipliers #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nIneqBorder, 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
                % try using a regular inverse first
                dir = dirsgn ...
                    * (M \ ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            catch
                dir = -(IMqrginv(M) * ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            end
            %# calculate derivative for rho*subgradient (Eq. 10) #%
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
            
            %# Misc. housekeeping #%
            % check for violations again
            signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (1 + 1e-8) & ...
                dirsgn*dir(1:nActive) <= (0 - 1e-8)  & ...
                betaPath(setActive, k-1) == 0);
            % Negative subgradient but positive derivative
            signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (0 + 1e-8) & ...
                (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
                betaPath(setActive, k-1) == 0);            
            
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
            % determine number of active/binding inequality constraints
            nIneqBorder = nnz(setIneqBorder);
            
            %# Recalculate derivative for coefficients & multipliers #%
            % construct matrix
            M = [H(setActive, setActive) Aeq(:,setActive)' ...
                A(setIneqBorder,setActive)'];
            M(end+1:end+m1+nIneqBorder, 1:nActive) = ...
                [Aeq(:,setActive); A(setIneqBorder,setActive)];
            % calculate derivative
            try
                % try using a regular inverse first
                dir = dirsgn ...
                    * (M \ ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            catch
                dir = -(IMqrginv(M) * ...
                    [subgrad(setActive); zeros(m1+nIneqBorder,1)]);
            end
            
            %# calculate derivative for rho*subgradient #%
            dirSubgrad = ...
                - [H(~setActive, setActive) Aeq(:,~setActive)' ...
                A(setIneqBorder,~setActive)'] * dir;
            
            %# Recheck for violations #%
            signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
                subgrad(setActive) <= (0 + 1e-8) & ...
                (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
                betaPath(setActive, k-1) == 0);
            % update violation counter
            violateCounter = violateCounter + 1;
            % break loop if needed
            if violateCounter >= maxiters
                break;
            end
            
        end
        
        
        %## update violation trackers to see if any issues persist ##%
        
        %# Inactive coefficients moving too slowly #%
        % Negative subgradient
        inactSlowNegIdx = ...
            find((1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (1*dirsgn + 1e-8) & ...
            1*dirsgn < dirSubgrad);
%         % Positive subgradient
        inactSlowPosIdx = find((-1*dirsgn - 1e-8) <= subgrad(~setActive) & ...
            subgrad(~setActive) <= (-1*dirsgn + 1e-8) & dirSubgrad < -1*dirsgn);
            
        %# "Active" coeficients estimated as 0 with potential sign mismatch #%
        % Positive subgrad but negative derivative
        signMismatchPosIdx = find((0 - 1e-8) <= subgrad(setActive) & ...
            subgrad(setActive) <= (1 + 1e-8) & ...
            dirsgn*dir(1:nActive) <= (0 - 1e-8) & ...
            betaPath(setActive, k-1) == 0);

        % Negative subgradient but positive derivative
        signMismatchNegIdx = find((-1 - 1e-8) <= subgrad(setActive) & ...
            subgrad(setActive) <= (0 + 1e-8) & ...
            (0 + 1e-8) <= dirsgn*dir(1:nActive) & ...
            betaPath(setActive, k-1) == 0);
        
        % break loop if needed
        if violateCounter >= maxiters
            break;
        end
    end % end of outer while loop
    
    % store number of violations
    violationsPath(k) = violateCounter;
    
    % calculate derivative for residual inequality  
    dirResidIneq = A(~setIneqBorder, setActive)*dir(1:nActive);
    
    
    %### Determine rho for next event (via delta rho) ###%
    %## Events based on coefficients changing activation status ##%
    
    % clear previous values for delta rho
    nextrhoBeta = inf(p, 1);
    
    %# Active coefficient going inactive #%
    nextrhoBeta(setActive) = -dirsgn*betaPath(setActive, k-1) ...
        ./ dir(1:nActive); 
    
    %# Inactive coefficient becoming positive #%
    t1 = dirsgn*rhoPath(k-1)*(1 - subgrad(~setActive)) ./ (dirSubgrad - 1);
    % threshold values hitting ceiling
    t1(t1 <= (0 + 1e-8)) = inf;

    %# Inactive coefficient becoming negative #%
    t2 = -dirsgn*rhoPath(k-1)*(1 + subgrad(~setActive)) ...
        ./ (dirSubgrad + 1);
    % threshold values hitting ceiling
    t2(t2 <= (0 + 1e-8)) = inf;        
        
    % choose smaller delta rho out of t1 and t2
    nextrhoBeta(~setActive) = min(t1, t2);
    % ignore delta rhos numerically equal to zero
    nextrhoBeta(nextrhoBeta <= 1e-8) = inf;

    
    %## Events based inequality constraints ##%
	% clear previous values
    nextrhoIneq = inf(m2, 1);
    
    %# Inactive inequality constraint becoming active #%
    nextrhoIneq(~setIneqBorder) = reshape(-dirsgn*residIneq(~setIneqBorder), ...
        nnz(~setIneqBorder), 1)./ reshape(dirResidIneq, nnz(~setIneqBorder), 1);    
    
    %# Active inequality constraint becoming deactive #%
    nextrhoIneq(setIneqBorder) = - dirsgn*dualpathIneq(setIneqBorder, k-1) ...
        ./ reshape(dir(nActive+m1+1:end), nIneqBorder,1);     
    % ignore delta rhos equal to zero
    nextrhoIneq(nextrhoIneq <= 1e-8) = inf;
   

    %# determine next rho ##
    % find smallest rho
    chgrho = min([nextrhoBeta; nextrhoIneq]);
    % find all indices corresponding to this chgrho
    idx = find(([nextrhoBeta; nextrhoIneq] - chgrho) <= 1e-8);
    
    % terminate path following if no new event found
    if isinf(chgrho)
        chgrho = rhoPath(k-1);
     end
    
    
    %## Update values at new rho ##%
    %# move to next rho #%
    % make sure next rho isn't negative
    if rhoPath(k-1) + dirsgn*chgrho < 0 
        chgrho = rhoPath(k-1);               
    end
    % calculate new value of rho
    rhoPath(k) = rhoPath(k-1) + dirsgn*chgrho;
    
    %# Update parameter and subgradient values #%
    % new coefficient estimates
    betaPath(setActive, k) = betaPath(setActive, k-1) ...
        + dirsgn*chgrho*dir(1:nActive);
    % force near-zero coefficients to be zero (helps with numerical issues)
    betaPath(abs(betaPath(:, k)) < 1e-12, k) = 0; 
    
    % new subgradient estimates
    subgrad(~setActive) = ...
        (rhoPath(k-1)*subgrad(~setActive) + ...
        dirsgn*chgrho*dirSubgrad)/rhoPath(k);
   
    %# Update dual variables #%
    % update lambda (lagrange multipliers for equality constraints)
    dualpathEq(:, k) = dualpathEq(:, k-1) ...
        + dirsgn*chgrho*reshape(dir(nActive+1:nActive+m1), m1, 1);
    % update mu (lagrange multipliers for inequality constraints)
    dualpathIneq(setIneqBorder, k) = dualpathIneq(setIneqBorder, k-1) ...
        + dirsgn*chgrho*reshape(dir(nActive+m1+1:end), nIneqBorder, 1);   
    
    % update residual inequality
    residIneq = A*betaPath(:, k) - b;
    
    
    %## update sets ##%
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
    
    % determine new number of active coefficients
    nActive = nnz(setActive);
    % determine number of active/binding inequality constraints
    nIneqBorder = nnz(setIneqBorder);
    
    %# Calcuate and store values of interest along the path #%
    % calculate value of objective function
    objValPath(k) = norm(y - X*betaPath(:, k))^2/2 + ...
        rhoPath(k)*sum(abs(betaPath(:, k)));
     
    % calculate degrees of freedom
    dfPath(k) = nActive - rankAeq - nIneqBorder;
    % break algorithm when df are exhausted
    if dfPath(k) >= n_orig
        break;
    end
end

% clean up output
warning(s);
warning(s2);
betaPath(:, k:end) = [];
rhoPath(k:end) = [];
objValPath(k:end) = [];
dfPath(k:end) = [];
dfPath(dfPath < 0) = 0; 
    
end



function [ IMqrginv ] = IMqrginv( A )
%IMqrginv Fast computation of a Moore-Penrose inverse
%   Source: Ataei (2014), "Improved Qrginv Algorithm for Computing 
%           Moore-Penrose Inverse Matrices"

% [m, n] = size(A);
[Q, R, P] = qr(A);
r = sum(any(abs(R) > 1e-05, 2));
Q = Q(:, 1:r);
R = R(1:r, :);

IMqrginv = P*(R'/(R*R')*Q');


end
