% read in data

clear;
fid = fopen('../../datasets/SAheart.data');
rawdata = textscan(fid, [repmat('%f ', 1, 5) '%s ' repmat('%f ', 1, 5)],...
    'HeaderLines', 1, 'delimiter', ',', 'CollectOutput', true);
fclose(fid);
rawdata = [rawdata{1,1} cellfun(@(s) strcmp(s,'Present'),rawdata{1,2}), ...
    rawdata{1,3}];

% sbp, tobacco, ldl, famhist, obesity, alcohol, age
X = rawdata(:,[2 3 4 6 8 9 10]);
X = bsxfun(@minus, X, mean(X,1));   % centering
X = bsxfun(@rdivide, X, std(X));    % standardize
X = [ones(size(X,1),1), X];
y = rawdata(:,end);
wt = ones(size(y));

%% fit logistic regression
b0 = glmfit(X,y,'binomial','constant','off');
[f,d1f,d2f,d3f] = glmfun(b0,X,y,wt,'binomial');
display(b0);
display(d1f);
display(d2f);

%% fused-lasso

Aeq = zeros(6,8);
Aeq(7:7:42) = 1;
Aeq(13:7:48) = -1;
beq = zeros(size(Aeq,1),1);

% % matlab constrained solver
% [b, logL] = fmincon(@(z) glmfun(z,X,y,wt,'poisson'), b0, [], [], Aeq, beq);
% display(b);
% display(logL);

% path algorithm

[x, rho_path, x_path, rho_kinks] = fminlin_path(@(z) glmfun(z,X,y,wt,'binomial'), ...
    b0, Aeq, beq, [], []);
display(x);
display(x_path);
display(rho_kinks);

% plot solution path

figure; hold on;
set(gca, 'FontSize', 20);
plot(rho_path',x_path(:,2:end));
title('fused-lasso path');
legend('sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age');
xlabel('\rho');
% print -depsc2 ../../manuscripts/notes/SAheart_fusepath.eps;

%% lasso

Aeq = zeros(7,8);
Aeq(8:8:56) = 1;
beq = zeros(size(Aeq,1),1);

% % matlab constrained solver
% [b, logL] = fmincon(@(z) glmfun(z,X,y,wt,'poisson'), b0, [], [], Aeq, beq);
% display(b);
% display(logL);

% path algorithm
[x, rho_path, x_path, rho_kinks] = fminlin_path(@(z) glmfun(z,X,y,wt,'binomial'), ...
    b0, Aeq, beq, [], []);
display(x);
display(x_path);
display(rho_kinks);

% plot solution path

figure; hold on;
set(gca, 'FontSize', 20);
plot(rho_path',x_path(:,2:end));
title('lasso path');
legend('sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age');
xlabel('\rho');
% print -depsc2 ../../manuscripts/notes/SAheart_lassopath.eps;

%% sparse fused-lasso

Aeq_fuse = zeros(6,8);
Aeq_fuse(7:7:42) = 1;
Aeq_fuse(13:7:48) = -1;
Aeq_lasso = zeros(7,8);
Aeq_lasso(8:8:56) = 1;
Aeq = [Aeq_fuse; Aeq_lasso];
beq = zeros(size(Aeq,1),1);

% % matlab constrained solver
% [b, logL] = fmincon(@(z) glmfun(z,X,y,wt,'poisson'), b0, [], [], Aeq, beq);
% display(b);
% display(logL);

% path algorithm

[x, rho_path, x_path, rho_kinks] = fminlin_path(@(z) glmfun(z,X,y,wt,'binomial'), ...
    b0, Aeq, beq, [], []);
display(x);
display(x_path);
display(rho_kinks);

% plot solution path

figure; hold on;
set(gca, 'FontSize', 20);
plot(rho_path',x_path(:,2:end));
title('fused-lasso path');
legend('sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age');
xlabel('\rho');
%print -depsc2 ../manuscripts/notes/SAheart_spfusepath.eps;