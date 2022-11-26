Dimension = 10;
mean = 1;
numberNodes = 10;
numQuad = 7;
numLin = 3;

rng('default');
B = normrnd(0,1,[Dimension,numLin]);
rng('default');
L = normrnd(0,1,[Dimension,1]);
Q = diag(abs(mean+L));

rng('default');
A1 = normrnd(0,1,[Dimension,Dimension,numberNodes]);
rng('default');
B1 = normrnd(0,1,[Dimension,numberNodes]);

cvx_begin
variables x(Dimension)
minimize (x'*Q*x)
cvx_end