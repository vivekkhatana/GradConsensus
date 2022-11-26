function [A, B, X, C, Lh, Lfhat, x_true] = Initialize_Test( Num_Agents, vec_Dim, Num_Examples, Training_Examples, index )

rand('seed', 0);
randn('seed', 0);


if index == 1
    for i = 1:Num_Agents
    
        a = normrnd(0, 1, [vec_Dim,vec_Dim]);
        A{i} = a'*a + eye(vec_Dim)*1e-2 ;
        Lf(i) = norm(A{i},2);
        B(:,i) = randn(vec_Dim,1);
        C(i) = randn(1);
        X(:,i) = 10*rand(vec_Dim,1);
%         B(:,i) = normrnd(0, 1, [vec_Dim,1]);
    end
%         C =  normrnd(0, 1, [Num_Agents,1]);
        Lh = max(Lf);
        Lfhat = 0.5*sum(Lf);
        
elseif index == 2
    
    x_true = randn(vec_Dim,1);
    
    for i = 1:Num_Agents
        A{i} = randn(Num_Examples,vec_Dim) ;
        B(:,i) = A{i}*x_true;
        X(:,i) = ones(vec_Dim,1);
        Lf(i) = norm(A{i}'*A{i},2);

    end
        C =  ones(Num_Agents,1);
        Lh = max(Lf);
        Lfhat = 0.5*sum(Lf);
        
elseif index == 3
    w = sprandn(vec_Dim, 1, (0.1*vec_Dim)/(vec_Dim));       % N(0,1), 10% sparse
    v = randn(1);                  % random intercept
    
    X0 = sprandn(Training_Examples*Num_Agents, vec_Dim, 10/vec_Dim);     % data / observations
    btrue = sign(X0*w);
%     btrue = sign(X0*w + v);
    
    % noise is function of problem size use 0.1 for large problem
    b0 = sign(X0*w + sqrt(0.1)*randn(Training_Examples*Num_Agents, 1)); % labels with noise
%     b0 = sign(X0*w + v + sqrt(0.1)*randn(Training_Examples*Num_Agents, 1)); % labels with noise
    
    % packs all observations into a (Training_Examples*Num_Agents \times vec_Dim) matrix
    A0 = spdiags(b0, 0, Training_Examples*Num_Agents, Training_Examples*Num_Agents)*X0;

%     ratio = sum(b0 == 1)/(Training_Examples*Num_Agents);
%     mu = 0.1*1/(Training_Examples*Num_Agents) * norm((1-ratio)*sum(A0(b0==1,:),1) + ratio*sum(A0(b0==-1,:),1), 'inf');
    
    x_true = w;
    
    for i = 1:Num_Agents
        X(:,i) = ones(vec_Dim,1);
        B(:,i) = b0(1+(i-1)*Training_Examples:i*Training_Examples,1);  
        A{i} = A0(1+(i-1)*Training_Examples:i*Training_Examples,:);  % make this a vector
        Lf(i) = norm(A{i}'*A{i},'fro');
    end   
        C =  ones(Num_Agents,1);
        Lh = max(Lf);
        Lfhat = 0.5*sum(Lf);
end

end

 