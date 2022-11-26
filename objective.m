function objval = objective(A,B,C,X,Num_Agents,index)
 objval = 0;
   if index == 1     
    for i = 1:Num_Agents
        objval = objval + 0.5*X(:,i)'*A{i}*X(:,i) + B(:,i)'*X(:,i) + C(i);
    end
   elseif index == 2
       for i = 1:Num_Agents
        objval = objval + 0.5*norm(A{i}*X(:,i) - B(:,i),2)^2;
       end
   elseif index == 3
       for i = 1:Num_Agents
           [sample_size,~] = size(A{i});
           z = X(:,i);
           for k = 1:sample_size
               var = 1 + exp( -B(k,i)*( A{i}(k,:)*z ) );
               innersum(k) = log(var);
           end
           obj(i) = sum(innersum);
       end
      objval = (sum(obj));
   end
end


 