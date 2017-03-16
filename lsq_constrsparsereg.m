function [betahat,stats] ...
    = lsq_constrsparsereg(X,y,lambda,varargin)
% LSQ_CONSTRSPARSEREG Sparse linear regression with constraints
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
%
% OPTIONAL NAME-VALUE PAIRS:
%
%   'A' - inequality constraint matrix
%   'b' - inequality constraint vector
%   'Aeq' - equality constraint matrix
%   'beq' - equality constraint vector
%   'admmAbsTol' - absolute tolerance for ADMM
%   'admmRelTol' - relative tolerance for ADMM
%   'admmMaxIter' - maximum number of iterations for ADMM
%   'admmScale' - ADMM scale parameter, 1/n is default
%   'admmVaryScale' - dynamically chance the ADMM scale parameter, 
%      false is default
%   'maxiter' - maximum number of iterations
%   'method' - 'cd' (default), 'qp' (quadratic programming, only for
%      lasso), or 'admm' (alternating direction method of multipliers)
%   'penidx' - a logical vector indicating penalized coefficients
%   'penparam' - index parameter for penalty; default values: ENET, 1,
%       LOG, 1, MCP, 1, POWER, 1, SCAD, 3.7
%   'pentype' - ENET|LOG|MCP|POWER|SCAD
%   'qp_solver' - 'matlab' (default), or 'GUROBI'
%   'sum_x_squares' - precomputed sum(wt*X.^2,1)
%   'weights' - a vector of prior weights
%   'x0' - a vector of starting point
%
% OUTPUT:
%   'betahat' - solution vector
%   'stats' - number of iterations (stats.qp_iters or stats.ADMM_iters)
%
% See also LSQ_SPARSEPATH,GLM_SPARSEREG,GLM_SPARSEPATH.
%
% Eexample
%
% References
%

% Copyright 2014-2017 University of California at Los Angeles and North Carolina State University
% Hua Zhou (huazhou@ucla.edu) and Brian Gaines (brgaines@ncsu.edu)

% input parsing rule
[n,p] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('y', @(x) length(y)==n);
argin.addRequired('lambda', @(x) x>=0);
argin.addParamValue('admmAbsTol', 1e-4, @(x) x>0);
argin.addParamValue('admmRelTol', 1e-4, @(x) x>0);
argin.addParamValue('admmMaxIter', 1e4, @(x) x>0);
argin.addParamValue('admmScale', 1/n, @(x) x>0);
argin.addParamValue('admmVaryScale', false, @islogical);
argin.addParamValue('A', [], @(x) size(x,2)==p);
argin.addParamValue('b', [], @(x) isnumeric(x));
argin.addParamValue('Aeq', [], @(x) size(x,2)==p);
argin.addParamValue('beq', [], @(x) isnumeric(x));
argin.addParamValue('qp_solver', 'matlab', @ischar);
argin.addParamValue('maxiter', 1000, @(x) isnumeric(x) && x>0);
argin.addParamValue('method', 'cd', @ischar);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', [], @isnumeric);
argin.addParamValue('penidx', true(p,1), @(x) islogical(x) && length(x)==p);
argin.addParamValue('projC', []);
argin.addParamValue('sum_x_squares', [], @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==p);
argin.addParamValue('weights', ones(n,1), @(x) isnumeric(x) && all(x>=0) && ...
    length(x)==n);
argin.addParamValue('x0', zeros(p,1), @(x) isnumeric(x) && length(x)==p);

% parse inputs
y = reshape(y,n,1);
argin.parse(X,y,lambda,varargin{:});
admmAbsTol = argin.Results.admmAbsTol;
admmRelTol = argin.Results.admmRelTol;
nADMM = argin.Results.admmMaxIter;
admmScale = argin.Results.admmScale;
admmVaryScale = argin.Results.admmVaryScale;
A = argin.Results.A;
b = argin.Results.b;
Aeq = argin.Results.Aeq;
beq = argin.Results.beq;
qp_solver = argin.Results.qp_solver;
maxiter = round(argin.Results.maxiter);
method = argin.Results.method;
penidx = reshape(argin.Results.penidx,p,1);
pentype = upper(argin.Results.penalty);
penparam = argin.Results.penparam;
projC = argin.Results.projC;
sum_x_squares = argin.Results.sum_x_squares;
wt = reshape(argin.Results.weights,n,1);
x0 = reshape(full(argin.Results.x0),p,1);

% check validity of qp_solver
if ~(strcmpi(qp_solver, 'matlab') || strcmpi(qp_solver, 'GUROBI'))
    error('sparsereg:lsq_sparsereg:qp_solver', ...
        'gp_solver not recognied');
end

% compute covariate norms if not supplied
if (isempty(sum_x_squares))
    sum_x_squares = sum(bsxfun(@times, wt, X.*X),1)';
else
    sum_x_squares = reshape(sum_x_squares,p,1);
end

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

% no linear constraints at all
if isempty(A) && isempty(b) && isempty(Aeq) && isempty(beq) && isempty(projC)
    
    % no penalization
    if abs(lambda) < 1e-16
        Xwt = bsxfun(@times, X, sqrt(wt));
        ywt = sqrt(wt).*y;
        betahat = Xwt\ywt;
        stats.qp_iters = 0;
        return;
    end
    
    % with penalization
    if strcmpi(method, 'cd')
        betahat = ...
            lsqsparse(x0,X,y,wt,lambda,sum_x_squares,penidx,...
            maxiter,pentype,penparam);
    elseif strcmpi(method, 'qp')
        % QP is only for lasso
        if ~(strcmpi(pentype,'enet') || ~strcmpi(pentype,'power')) ...
                || penparam~=1
            error('sparsereg:lsq_sparsereg:qp_notapply', ...
                'Quadratic programming is only for solving lasso');
        end
        % set up QP problem
        Xwt = bsxfun(@times, X, sqrt(wt));
        ywt = sqrt(wt).*y;
        H = Xwt'*Xwt;   % quadratic coefficient
        H = [H -H; -H H];
        f = - Xwt'*ywt; % linear coefficient
        f = [f; -f] + lambda*[penidx; penidx];
        lb = zeros(2*p,1);
        ub = inf(2*p,1);
        if strcmpi(qp_solver, 'GUROBI')
            % use QUROBI solver if possible
            gmodel.obj = f;
            gmodel.A = sparse(zeros(0,2*p));
            gmodel.sense = '=';
            gmodel.rhs = zeros(0,1);
            gmodel.lb = lb;
            gmodel.Q = sparse(H)/2;
            gparam.OutputFlag = 0;
            gresult = gurobi(gmodel, gparam);
            betahat = gresult.x(1:p) - gresult.x(p+1:end);
            stats.qp_iters=gresult.baritercount; % store gurobi iters
            stats.qp_objval=gresult.objval; % store gurobi obj. value
        elseif strcmpi(qp_solver, 'matlab')
            % use matlab quadprog()
            options.Algorithm = 'interior-point-convex';
            options.Display = 'off';
            [x,fval,~,output, dual_vars] = quadprog(H, f, [], [], [], [], lb, ub, ...
                [max(x0,0);min(x0,0)], options);
            betahat = x(1:p) - x(p+1:end);
            stats.qp_iters=output.iterations; % store matlab QP iters
            stats.qp_objval=fval; % store QP objective value
            stats.qp_dualEq = dual_vars.eqlin;
            stats.qp_dualIneq = dual_vars.ineqlin;
        end
    end
    
    
else    % with linear constraints
    
    if strcmpi(method, 'qp')
        
        % QP is only for lasso
        if ~(strcmpi(pentype,'enet') || ~strcmpi(pentype,'power')) ...
                || penparam~=1
            error('sparsereg:lsq_sparsereg:qp_notapply', ...
                'Quadratic programming is only for solving lasso');
        end
        % set up QP problem
        Xwt = bsxfun(@times, X, sqrt(wt));
        ywt = sqrt(wt).*y;
        H = Xwt'*Xwt;   % quadratic coefficient
        H = [H -H; -H H];
        f = - Xwt'*ywt; % linear coefficient
        f = [f; -f] + lambda*[penidx; penidx];
        lb = zeros(2*p,1);  % boundary condition
        ub = inf(2*p,1);
        if strcmpi(qp_solver, 'GUROBI')
            % use GUROBI solver if available
            gmodel.obj = f;
            gmodel.A = sparse([Aeq, -Aeq; A, -A]);
            gmodel.sense = ...
                [repmat('=', size(Aeq,1), 1); repmat('<', size(A,1), 1)];
            gmodel.rhs = [beq; b];
            gmodel.lb = lb;
            gmodel.ub = ub;
            gmodel.Q = sparse(H)/2;
            gparam.OutputFlag = 0;
            gresult = gurobi(gmodel, gparam);
            betahat = gresult.x(1:p) - gresult.x(p+1:end);
            stats.qp_iters=gresult.baritercount; % store gurobi iters
            stats.qp_objval=gresult.objval; % store gurobi obj. value
            stats.qp_dualEq = gresult.pi(1:size(Aeq,1));
            stats.qp_dualIneq = gresult.pi(size(Aeq,1)+1:end);
            %         dualpathEq(:,1) = gresult.pi(m2+1:end);
%         dualpathIneq(:,1) = reshape(gresult.pi(1:m2), m2, 1);
        elseif strcmpi(qp_solver, 'matlab')
            % use matlab quadprog()
            options.Algorithm = 'interior-point-convex';
            options.Display = 'off';
            [x,fval,~,output, dual_vars] = quadprog(H, f, [A, -A], b, [Aeq, -Aeq], beq, ...
                lb, ub, [max(x0,0);min(x0,0)], options);
            betahat = x(1:p) - x(p+1:end);
            stats.qp_iters=output.iterations; % store matlab QP iters
            stats.qp_objval=fval; % store QP objective value
            stats.qp_dualEq = dual_vars.eqlin;
            stats.qp_dualIneq = dual_vars.ineqlin;
        end
        
    elseif strcmpi(method, 'ADMM')
        
        if isempty(projC)
            % set up dual QP problem for projection to polyhedra
            G = [Aeq; A];
            H = G*G';
            lb = zeros(size(G,1), 1);
            lb(1:size(Aeq,1)) = -inf;
            ub = inf(size(G,1), 1);
            if strcmpi(qp_solver, 'GUROBI')
                % use GUROBI solver if available
                gmodel.A = sparse(zeros(0,size(G,1)));
                gmodel.sense = repmat('=',0,1);
                gmodel.rhs = zeros(0,1);
                gmodel.lb = lb;
                gmodel.ub = ub;
                gmodel.Q = sparse(H)/2;
                gparam.OutputFlag = 0;
            elseif strcmpi(qp_solver, 'matlab')
                % use matlab quadprog()
                options.Algorithm = 'interior-point-convex';
                %options.Algorithm = 'active-set';
                options.Display = 'off';
            end
        end
        
        % initialize
        betahat = x0;
        if isempty(projC)
            constrRes = G*betahat - [beq;b];
            if strcmpi(qp_solver, 'GUROBI')
                gmodel.obj = -constrRes;
                gresult = gurobi(gmodel, gparam);
                x = gresult.x;
            elseif strcmpi(qp_solver, 'matlab')
                x = quadprog(H, -constrRes, [], [], [], [], ...
                    lb, ub, [], options);
            end
            z = x0 - G'*x;
        else
            z = projC(betahat);
        end
        u = betahat - z;
        
        % ADMM loop
        for iADMM = 1:nADMM
            % update beta - lasso
            betahat = ...
                lsqsparse(betahat,[X; eye(p)/sqrt(admmScale)], ...
                [y;(z-u)/sqrt(admmScale)], ...
                [wt;ones(p,1)],lambda,sum_x_squares+1/admmScale,penidx,...
                maxiter,pentype,penparam);
%             betahat = lsq_constrsparsereg([X; eye(p)/sqrt(admmScale)], ...
%                 [y;(z-u)/sqrt(admmScale)], lambda, 'method', 'qp', ...
%                 'qp_solver', 'GUROBI');
%             betahat = lasso([X; eye(p)/sqrt(admmScale)], ...
%                 [y;(z-u)/sqrt(admmScale)], 'Lambda', lambda/(size(X,1)+p), ...
%                 'Standardize', false);
            % update z - projection to constraint set
            v = betahat + u;
            zOld = z;
            if isempty(projC)
                % project to polyhedron using quadratic programming
                constrRes = G*v - [beq;b];
                if strcmpi(qp_solver, 'GUROBI')
                    gmodel.obj = - constrRes;
                    gresult = gurobi(gmodel, gparam);
                    x = gresult.x;
                elseif strcmpi(qp_solver, 'matlab')
                    x = quadprog(H, - constrRes, [], [], [], [], ...
                        lb, ub, [], options);
                end
                z = v - G'*x;
            else
                % project using user-supplied projection
                z = projC(v);
            end
            % update scaled dual variables u
            dualResNorm = norm((z-zOld)/admmScale);
            primalRes = betahat - z;
            primalResNorm = norm(primalRes);
            u = u + primalRes;
            % stopping criterion
            %display([primalResNorm dualResNorm]);
            %display(sum(z));
            if (primalResNorm <= sqrt(p)*admmAbsTol ...
                    + admmRelTol*max(norm(betahat),norm(z))) && ...
                    (dualResNorm <= sqrt(n)*admmAbsTol ...
                    + admmRelTol*norm(u/admmScale))
                break;
            end
            % update ADMM scale parameter if requested
            if admmVaryScale
                if primalResNorm/dualResNorm>10
                    admmScale = admmScale/2;
                    u = u/2;
                elseif primalResNorm/dualResNorm<0.1
                    admmScale = admmScale*2;
                    u = 2*u;
                end
            end
            
        end
        stats.ADMM_iters = iADMM;
        
    end
    
end

end