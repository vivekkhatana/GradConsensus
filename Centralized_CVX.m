function [xstar, fstar] = Centralized_CVX(vec_Dim, A, B, C, V, Num_Agents, objindex)

cvx_begin quiet
variables x(vec_Dim)

objective_func = 0;

   if objindex == 1     
        for i = 1:Num_Agents
            objective_func = objective_func + 0.5*x'*A{i}*x + B(:,i)'*x + C(i);
        end
   elseif objindex == 2
       for i = 1:Num_Agents
        objective_func = objective_func + 0.5*power(2,norm(A{i}*x - B(:,i),2));
       end
   elseif objindex == 3
       for i = 1:Num_Agents
           [~,sample_size] = size(A{i});
           for j = 1:sample_size
               innersum(i) = (1/sample_size)*( log( 1 + exp( -B(j,i)*( A{i}(:,j)'*x + V(i) ) ) ) );
           end
       end
       objective_func = sum(innersum);
   end
   
minimize objective_func
cvx_end
xstar = x.*ones(vec_Dim,Num_Agents);
fstar = objective(A,B,C,xstar,V,Num_Agents,objindex);
end
