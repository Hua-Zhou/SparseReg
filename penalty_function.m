function [pen,d1pen,d2pen,dpendlambda] = ...
    penalty_function(beta,lambda,pentype,penparam)
% PENALTY_FUNCTION The penalty value and derivatives of penalties
%   The outputs have same size as beta   
%
% INPUT:
%   beta - regression coefficients
%   lambda - penalty constant    
%   penname - 'enet'|'log'|'scad'|'mcp'|'bridge'
%   penargs - optional arguments for penalty function penname
%
% Output:
%   pen - p-by-1 penalty values
%   d1pen - p-by-1 first derivatives
%   d2pen - p-by-1 second derivatives
%   dpendlambda - p-by-1 second mixed derivatives
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou, hua_zhou@ncsu.edu
% RELEASE DATE: ??/??/????

% check arguments
if (lambda<0)
    error('penalty constant lambda should be nonnegative');
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
[pen,d1pen,d2pen,dpendlambda] = ...
    penalty(beta,lambda,pentype,penparam);

end