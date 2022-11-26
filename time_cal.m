% This code is written for simulations of gradient consensus (journal
% version) in case of strongly convex functions

clear all

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
Diameter = 14;

% numGraphs = 100;
% numIniCond = 100;

%% Choose the objective function
index = 1; % index = 1 :Strongly Conv (Quadratic Prob), index = 2 :Conv(Least Squares), index = 3: Logistic Reg 
convergence_Tol = 0.01;

%%% Quadratic Optimization
Dimension = 10; % , numberNodes  

%%% Least Squares
NumofFeatures = 10; % , numberNodes
Num_Examples = 100;

%%% Logistic Regression
Training_Examples = 150;
Num_Features = 15;
Num_Subsystems = 100;
sample_size = 10;


vec_Dim = NumofFeatures;
%% Problem data initialization
[A, B, IniEstimate, C, V, Lh, Lfhat] = Initialize_Test( numberNodes, vec_Dim, Num_Examples, sample_size, index );


%% CVX code to solve the problem centrally

[xstar, fstar] = Centralized_CVX(vec_Dim, A, B, C, V, numberNodes, index);
% xstar = [0.3920,0.1139,-0.3212,0.0373,-0.0615,-0.0294,-0.1725,-0.1359,0.0639,0.0813]';
Xstar = xstar.*ones(vec_Dim,numberNodes);
% fstar = 0;

%% Distributed implementation
stepsize = [0.1, 0.002, 0.001, 0.0005, 0.0003];
alpha = stepsize(4);
Max_Iter = 500*Diameter;
% rho = rho_max(numberNodes, alpha, convergence_Tol, Lh, Lfhat)/2;
rho = 0.01;

%% Calculating time required to run a function.
%The timeit function calls the specified function multiple times, and returns the median of the measurements.

f_EXTRA = @()EXTRA(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
t_EXTRA = timeit(f_EXTRA)

f_DGD = @()DGD(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
t_DGD = timeit(f_DGD)

f_gradConsensus = @()gradConsensus(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, IniEstimate, C, V, Diameter, ...
                                                   rho, alpha, index, Xstar, fstar,Max_Iter);
t_gradConsensus = timeit(f_gradConsensus)

f_PushPull = @()PushPull(vec_Dim, numberNodes, Weight_Row_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
                                                                                          
t_PushPull = timeit(f_PushPull) 

%% Calculating the CPU time

itime_EXTRA = cputime;
EXTRA(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
ftime_EXTRA = cputime - itime_EXTRA

itime_DGD = cputime;
DGD(vec_Dim, numberNodes, Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
ftime_DGD = cputime - itime_DGD

itime_gradConsensus = cputime;
gradConsensus(vec_Dim, numberNodes,Weight_Doub_Stoc, A, B, IniEstimate, C, V, Diameter, ...
                                                   rho, alpha, index, Xstar, fstar,Max_Iter);
ftime_gradConsensus = cputime - itime_gradConsensus

itime_PushPull  = cputime;
PushPull(vec_Dim, numberNodes, Weight_Row_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter);
                                                                                       
ftime_PushPull = cputime - itime_PushPull 

                                                                                          
