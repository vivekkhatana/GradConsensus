function [history_cons_res,history_sol_res,history_objval_res,comm,sum_avg_grad,ftime] = nearDGD(vec_Dim, numberNodes,...
                                                      Weight_Row_Stoc, A, B, IniEstimate, C,...
                                                      alpha, index, Xstar, fstar,Max_Iter)
        

               
        history_sol_res = zeros(1,Max_Iter+1);
        history_cons_res = zeros(1,Max_Iter+1);
        history_objval_res = zeros(1,Max_Iter+1);        
        
        X = zeros(vec_Dim,numberNodes,Max_Iter+1);
        Z = zeros(vec_Dim,numberNodes,Max_Iter+1);
        X(:,:,1)= IniEstimate;
        Z(:,:,1)= IniEstimate;
        
        Iterdiff = 1;
        sum_avg_grad = zeros(Max_Iter,1);
  
%         tic 
        comm = [];

        
        for i = 2:2:Max_Iter
               tic;
               
               grad = gradient( A, B, X(:,:,i-1), numberNodes, index);
               for j = 1:numberNodes
                  X(:,j,i) = X(:,j,i-1) - alpha*grad(:,j);
               end
%                if (mod(i,300) == 0)
%                 Iterdiff = 2*Iterdiff;
%                end
               Iterdiff = i;
               Y = X(:,:,i);
               t = Iterdiff;
               while t > 0
                   Y = Y*Weight_Row_Stoc;
                   t = t - 1;
               end
               
               Z(:,:,i) = Y;     
                              
               ftime(i) = toc; 
               itime = 0;
               
%                [Z(:,:,i),Iterdiff] = Consensus(Weight_Row_Stoc,Diameter,X(:,:,i),rho);
               
               comm(i) = Iterdiff;
               
               avg_X = mean(Z(:,:,i),2);
               Avg_X = avg_X.*ones(vec_Dim,numberNodes);
               avg_grad = gradient( A, B, Avg_X, numberNodes, index);
               sum_avg_grad(i) = norm(sum(avg_grad,2),2);
               
               var_sol = X(:,:,i) - Xstar;
               var_avgsol = Z(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;
               history_sol_res(i) = norm(var_sol,2)^2/norm(var_inisol,2)^2;

               history_cons_res(i) =  norm(Z(:,:,i) - Avg_X);

%                history_cons_res(i) = norm(var_avgsol,2)^2/norm(var_inisol,2)^2;
               history_objval_res(i) = abs(objective(A,B,C,Avg_X,numberNodes,index) - fstar);
               
%                 if history_sol_res(i) < 1e-3
%                    break;
%                 end
                
               X(:,:,i+1) = Z(:,:,i); 
        end
%         time = toc;
%         commpertime = sum(Iterdiff)/time; 
%     itime = 0;
end 