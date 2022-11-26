function grad = gradient( A, B, X, Num_Agents, index)
 grad = [];
  if index == 1
      for j = 1:Num_Agents
        grad(:,j) = A{j}*X(:,j) + B(:,j);
      end      
  elseif index == 2
      for j = 1:Num_Agents
        grad(:,j) = A{j}'*(A{j}*X(:,j) - B(:,j));
      end 
  elseif index == 3
      for j = 1:Num_Agents
           [sample_size,vec_dim] = size(A{j});
           z = X(:,j);
           for k = 1:sample_size
               var = 1 + exp( -B(k,j)*( A{j}(k,:)*z ) );
               var1 = exp( -B(k,j)*( A{j}(k,:)*z ) )*(-B(k,j)*( A{j}(k,:))); 
%                var2 = exp( -B(k,j)*( A{j}(k,:)*z(2:end) + z(1) ) )*(-B(k,j)); 
               innergrad(:,k) = (1/var)*var1';
           end
           grad(:,j) = sum(innergrad,2);
      end
  end
end

 