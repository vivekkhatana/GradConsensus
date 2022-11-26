function rho_max = rho_max( Num_Agents, alpha, convergence_Tol, Lh, Lfhat )
  var1 = (Num_Agents + 1)*Num_Agents^2*Lh^2*alpha + Num_Agents*alpha + (Num_Agents + 1)*Num_Agents*Lfhat;
  var2 = alpha/var1;
  var3 = sqrt(var2);
  rho_max = convergence_Tol*var3;
end

 