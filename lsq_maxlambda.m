function [maxlambda] = lsq_maxlambda(a,b,pentype,penparam)
% LSQ_MAXLAMBDA Max lambda for the solution path of sparse linear regression
%   MAXLAMBDA = LSQ_MAXLAMBDA(A,B,PENTYPE,PENPARAM) returns the max lambda
%   of solution path for penalized lienar regression. A is a vector of
%   quadratic coefficients. B is a vector of linear coefficients. PENTYPE
%   is the penalty name 'enet', 'log', 'mcp', 'power', or 'scad'. PENPARAM
%   is the parameter for the penalty. Allowable range and default values of
%   PENPARAM are enet [1,2] (1 by default), log (0,inf) (1 by default), mcp
%   (0,inf) (1 by default), power (0,2] (1 by default), scad (2,inf) (3.7
%   by default). The result MAXLAMBDA is the max lambda such that argmin
%   0.5*a*x^2+b*x+penalty(abs(x),lambda) is nonzero.
%
%   See also LSQ_SPARSEREG,LSQ_SPARSEPATH.
%
%   References:
%
% Copyright 2017 University of California at Los Angeles
% Hua Zhou (huazhou@ucla.edu)

% check proper input arguments
[n,m] = size(a);
if (size(b,1)~=n || size(b,2)~=m)
    error('a and b have incompatible sizes');
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
    error('penalty type not recogonized. ENET|LOG|MCP|POWER|SCAD accepted');
end

% call the mex function
maxlambda = lsqmaxlambda(a,b,pentype,penparam);

end