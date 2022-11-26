function [history_cons_res,history_sol_res,history_objval_res,sum_avg_grad,ftime] = EXTRA(vec_Dim, numberNodes,...
                                                      Weight_Doub_Stoc, A, B, IniEstimate, C, V, ...
                                                      alpha, index, Xstar, fstar,Max_Iter)

       
        
        history_sol_res = zeros(1,Max_Iter+1);
        history_cons_res = zeros(1,Max_Iter+1);
        history_objval_res = zeros(1,Max_Iter+1);        
        
        X = zeros(vec_Dim,numberNodes,Max_Iter+1);
        X(:,:,1)= IniEstimate;
        sum_avg_grad = zeros(Max_Iter,1);
        
        X(:,:,2)= X(:,:,1)*Weight_Doub_Stoc - alpha*gradient( A, B, X(:,:,1), numberNodes,index);            
        Weight_Doub_Stoc_tilde = 0.5*( eye(length(Weight_Doub_Stoc)) + Weight_Doub_Stoc );        
        
        comm = [];

        
        for i = 2:1:Max_Iter
               tic;
            
               var = gradient(A, B, X(:,:,i), numberNodes,index) - gradient(A, B, X(:,:,i-1), numberNodes,index);
               X(:,:,i+1) = X(:,:,i)*2*Weight_Doub_Stoc_tilde - X(:,:,i-1)*Weight_Doub_Stoc_tilde - alpha*var;
               
               ftime(i) = toc;    
               itime = 0;               
               
               avg_X = mean(X(:,:,i),2);
               Avg_X = avg_X.*ones(vec_Dim,numberNodes);
               avg_grad = gradient( A, B, Avg_X, numberNodes, index);
               sum_avg_grad(i) = norm(sum(avg_grad,2),2);
                            
               var_sol = X(:,:,i) - Xstar;            
               var_inisol = IniEstimate - Xstar;
               history_sol_res(i) = norm(var_sol,2)^2/norm(var_inisol,2)^2;
               
               avg_X = mean(X(:,:,i),2);
               Avg_X = avg_X.*ones(vec_Dim,numberNodes);
               history_cons_res(i) = 0;
               history_cons_res(i) =  norm(X(:,:,i) - Avg_X);
               
               
               history_objval_res(i) = abs(objective(A,B,C,X(:,:,i),V,numberNodes,index) - fstar);
               
%                if history_sol_res(i) < 1e-5
%                    comm = i;
%                    break;
%                end

        end
  
end

 