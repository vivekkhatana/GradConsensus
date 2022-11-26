% This code is written for simulations of gradient consensus (journal version) 

% clear all;

%% graph initialization
% load('graphArray_20Nodes_10000ConnectedDiGraph');
load('graphArray_100Nodes_1_D14ConnectedDiGraph');
Num_graphs = size(arr,1);
numberNodes = sqrt(size(arr,2));
graphNo = 1;
currentG = arr(graphNo,:);   
currentG = reshape(currentG,numberNodes,numberNodes)'+eye(numberNodes);
% row stochastic for vector updates X(k+1) = X(k)*P
Weight_Row_Stoc = PRowStochastic(currentG,numberNodes); 
Weight_Doub_Stoc = PDoubleStochastic(currentG,numberNodes);
Diameter = 5;

% numGraphs = 100;
% numIniCond = 100;

%% Choose the objective function
index = 3; % index = 1 :Strongly Conv (Quadratic Prob), index = 2 :Conv(Least Squares), index = 3: Logistic Reg 
convergence_Tol = 0.01;

%%% Quadratic Optimization
Dimension = 10; % , numberNodes  

%%% Least Squares
NumofFeatures = 10; % , numberNodes
Num_Examples = 100;

%%% Logistic Regression
Training_Examples = 150;
Num_Features = 15;


vec_Dim = NumofFeatures;
%% Problem data initialization
[A, B, IniEstimate, C, Lh, Lfhat, x_true] = Initialize_Test( numberNodes, vec_Dim, Num_Examples, Training_Examples, index );


%% CVX code to solve the problem centrally

Xstar = x_true.*ones(vec_Dim,numberNodes);
fstar = objective(A,B,C,Xstar,numberNodes,index);

%% Distributed implementation
stepsize = [0.1, 0.002, 0.001, 0.0005, 0.0003];
alpha1 = stepsize(2);
alpha2 = stepsize(4);
Max_Iter = 50*Diameter;
% rho = rho_max(numberNodes, alpha, convergence_Tol, Lh, Lfhat)/2;
[history_cons_res_GradCons,history_sol_res_GradCons,history_objval_err_GradCons,comm,sumgrad,time_GradCons] = gradConsensus(vec_Dim, numberNodes,...
                                                                                              Weight_Doub_Stoc, A, B, IniEstimate, C, Diameter, ...
                                                                                              alpha1, index, Xstar, fstar,Max_Iter);
                                                                                          
% [history_cons_res_DGD,history_sol_res_DGD,history_objval_err_DGD,sumgrad_DGD,time_DGD] = DGD(vec_Dim, numberNodes,...
%                                                       Weight_Doub_Stoc, A, B, IniEstimate, C,...
%                                                       alpha2, index, Xstar, fstar,10*Max_Iter);
                                                  

% [history_cons_res_EXTRA,history_sol_res_EXTRA,history_objval_err_EXTRA,sumgrad_EXTRA, time_EXTRA] = EXTRA(vec_Dim, numberNodes,...
%                                                       Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
%                                                       alpha, index, Xstar, fstar,Max_Iter);
%                                                   
% [history_cons_res_nearDGD,history_sol_res_nearDGD,history_objval_err_nearDGD,comm_nearDGD,sumgrad_nearDGD, time_nearDGD] = nearDGD(vec_Dim, numberNodes,...
%                                                       Weight_Row_Stoc, A, B, IniEstimate, C,...
%                                                       alpha2, index, Xstar, fstar,Max_Iter);
%  hh=zeros(1,floor(Max_Iter/2));
%             hh(1)=1;
%             for i = 2:2:Max_Iter
%                 hh(i/2+1) = hh(i/2)+i/2+Iterdiff(i);
%             end
%%%  Plotting for data


%% Plotting 
flen = floor(length(history_objval_err_GradCons)/2);
slen = floor(length(history_sol_res_GradCons)/2);
aslen = floor(length(history_cons_res_GradCons)/2);

% figure(1);
% plot(1:flen, history_objval_err_GradCons(2:2:length(history_objval_err_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth', 1);
% hold on 
% plot(1:flen, history_objval_err_DGD(2:2:length(history_objval_err_DGD)),'o', 'MarkerFaceColor', 'b', 'MarkerSize', 1, 'LineWidth', 1);
% hold on
% plot(1:flen, history_objval_err_EXTRA(2:2:length(history_objval_err_EXTRA)),'v', 'MarkerFaceColor', 'k', 'MarkerSize', 1, 'LineWidth', 1);
% hold on
% plot(1:flen, history_objval_err_PushPull(2:2:length(history_objval_err_PushPull)),'<', 'MarkerFaceColor', 'r', 'MarkerSize', 1, 'LineWidth', 1);
% legend('PushPull, \alpha = 0.0005','DGD , \alpha = 0.0005 ','EXTRA, \alpha = 0.0005 ','GradConsensus, \alpha = 0.0005', 'interpreter', 'latex', 'Location','NorthEast')
% ylabel('f(k)-f^*'); xlabel('iter (k)');
 

% figure(2);
% plot(1:slen, history_sol_res_GradCons(2:2:length(history_sol_res_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth',1);
% hold on 
% plot(1:slen, history_sol_res_DGD(2:2:length(history_sol_res_DGD)),'o', 'MarkerFaceColor', 'b', 'MarkerSize', 1, 'LineWidth', 1);
% hold on 
% plot(1:slen, history_sol_res_EXTRA(2:2:length(history_sol_res_EXTRA)),'v','MarkerFaceColor', 'k', 'MarkerSize', 1, 'LineWidth', 1);
% hold on
% plot(1:slen, history_sol_res_PushPull(2:2:length(history_sol_res_PushPull)),'<', 'MarkerFaceColor', 'r', 'MarkerSize', 1, 'LineWidth', 1);
% legend('PushPull, \alpha = 0.0005','DGD , \alpha = 0.0005 ','EXTRA, \alpha = 0.0005','GradConsensus, \alpha = 0.0005', 'interpreter', 'latex', 'Location','NorthEast')
% ylabel('Residual'); xlabel('iter (k)');
% % ylabel('$||U(k) - U^*||^2/||U(0) - U^*||^2$','interpreter','latex'); xlabel('iter (k)','interpreter','latex');
aslen1 = floor(length(history_sol_res_DGD)/2);

figure(3);
plot(1:aslen, (1./(1:aslen)).*history_sol_res_GradCons(2:2:length(history_sol_res_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth', 1);
hold on 
plot(1:aslen1, history_sol_res_DGD(2:2:length(history_sol_res_DGD)),'o','MarkerFaceColor', 'b', 'MarkerSize', 1, 'LineWidth', 1);
hold on 
% plot(1:aslen, history_cons_res_EXTRA(2:2:length(history_cons_res_EXTRA)),'v','MarkerFaceColor', 'k', 'MarkerSize', 1, 'LineWidth', 1);
% hold on
plot(1:aslen, history_sol_res_nearDGD(2:2:length(history_sol_res_nearDGD)),'<', 'MarkerFaceColor', 'r', 'MarkerSize', 1, 'LineWidth', 1);
legend('GradConsensus','DGD','nearDGD', 'interpreter', 'latex', 'Location','NorthEast')
% ylabel('Total Agent Mismatch'); 
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('# Gradient Computations (k)');
% ylabel('$||\hat{U}(k) - U^*||^2/||U(0) - U^*||^2$','interpreter','latex'); xlabel('iter (k)','interpreter','latex');

t_GradCons = cumsum(time_GradCons);
t_DGD = cumsum(time_DGD);
t_nearDGD = cumsum(time_nearDGD);

figure(4);
plot(t_GradCons(1:2:length(t_GradCons)), (1./(1:aslen)).*history_sol_res_GradCons(2:2:length(history_sol_res_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth',1);
hold on 
plot(t_DGD(1:2:length(t_DGD)), history_sol_res_DGD(2:2:length(history_sol_res_DGD)),'--','MarkerFaceColor','r', 'MarkerSize', 1, 'LineWidth',1);
hold on 
plot(t_nearDGD(1:2:length(t_nearDGD)), history_sol_res_nearDGD(2:2:length(history_sol_res_nearDGD)),'--','MarkerFaceColor','b', 'MarkerSize', 1, 'LineWidth',1);
legend('GradConsensus','DGD','nearDGD', 'interpreter', 'latex', 'Location','NorthEast')
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
% ylabel('Total Agent Mismatch');
xlabel('time (sec)');

comm_GradCons = 0.5*cumsum(comm);
comm_nearDGD1 = cumsum(comm_nearDGD);
% comm1 = comm(2:2:length(comm_GradCons));
% comm_nearDGD2 = comm_nearDGD(2:2:length(comm_nearDGD1));
figure(6);
plot(comm_GradCons(2:2:length(comm_GradCons)), history_sol_res_GradCons(2:2:length(history_sol_res_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth',1);
hold on 
plot(1:aslen1, history_sol_res_DGD(2:2:length(history_sol_res_DGD)),'o','MarkerFaceColor', 'b', 'MarkerSize', 1, 'LineWidth', 1);
hold on
plot(comm_nearDGD1(2:2:length(comm_nearDGD1)), history_sol_res_nearDGD(2:2:length(history_sol_res_nearDGD)),'--','MarkerFaceColor','r', 'MarkerSize', 1, 'LineWidth',1);
legend('GradConsensus','DGD','nearDGD', 'interpreter', 'latex', 'Location','NorthEast')
% ylabel('Total Agent Mismatch');
ylabel('$\displaystyle \frac{\|\mathbf{x}^k - \mathbf{x}^*\|^2}{\|\mathbf{x}^0 - \mathbf{x}^*\|^2}$','interpreter','latex'); 
xlabel('# Communication Steps'); 

figure(7);
plot(1:length(comm_GradCons), comm_GradCons(1:1:length(comm_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth',1);
hold on 
plot(1:length(comm_nearDGD1), comm_nearDGD1(1:1:length(comm_nearDGD1)),'--','MarkerFaceColor','r', 'MarkerSize', 1, 'LineWidth',1);
legend('GradConsensus','nearDGD', 'interpreter', 'latex', 'Location','NorthEast')
ylabel('Total Communication Steps'); 
xlabel('Algorithm Iterations');

figure(8);
plot(1:aslen, (1./(1:aslen)).*history_cons_res_GradCons(2:2:length(history_sol_res_GradCons)),'--','MarkerFaceColor','g', 'MarkerSize', 1, 'LineWidth', 1);
hold on 
plot(1:aslen1, history_cons_res_DGD(2:2:length(history_sol_res_DGD)),'o','MarkerFaceColor', 'b', 'MarkerSize', 1, 'LineWidth', 1);
hold on 
% plot(1:aslen, history_cons_res_EXTRA(2:2:length(history_cons_res_EXTRA)),'v','MarkerFaceColor', 'k', 'MarkerSize', 1, 'LineWidth', 1);
% hold on
plot(1:aslen, history_cons_res_nearDGD(2:2:length(history_sol_res_nearDGD)),'<', 'MarkerFaceColor', 'r', 'MarkerSize', 1, 'LineWidth', 1);
legend('GradConsensus','DGD','nearDGD', 'interpreter', 'latex', 'Location','NorthEast')
% ylabel('Total Agent Mismatch'); 
ylabel('Total Agent Mismatch'); 
xlabel('Total Iterations');