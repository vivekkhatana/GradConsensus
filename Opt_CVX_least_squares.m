Dimension = 10;
numberNodes = 10;
% load('s');
% rng(s)
rng('default');
A1 = normrnd(0,1,[Dimension,Dimension,numberNodes]);
rng('default');
B1 = normrnd(0,1,[Dimension,numberNodes]);

cvx_begin
variables x(numberNodes)
minimize (0.5*(power(2,norm(A1(:,:,1)*x-B1(:,1),2))+power(2,norm(A1(:,:,2)*x-B1(:,2),2))+power(2,norm(A1(:,:,3)*x-B1(:,3),2))+power(2,norm(A1(:,:,4)*x-B1(:,4),2))+power(2,norm(A1(:,:,5)*x-B1(:,5),2))+power(2,norm(A1(:,:,6)*x-B1(:,6),2))+power(2,norm(A1(:,:,7)*x-B1(:,7),2))+power(2,norm(A1(:,:,8)*x-B1(:,8),2))+power(2,norm(A1(:,:,9)*x-B1(:,9),2))+power(2,norm(A1(:,:,10)*x-B1(:,10),2))))
cvx_end