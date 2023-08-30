#helper functions
deno_integrand = function(x, B, tau_squared){
  tau = sqrt(tau_squared)
  return((sqrt(x) - tau)*dchisq(x, B))
}

deno_integral = function(B, tau_squared){
  return(integrate(deno_integrand, lower = tau_squared, upper = Inf, B, tau_squared)$value)
}

num_integrand = function(x, B, tau_squared){
  tau = sqrt(tau_squared)
  return((sqrt(x) - tau)^2*dchisq(x, B))
}

num_integral = function(B, tau_squared){
  return(integrate(num_integrand, lower = tau_squared, upper = Inf, B, tau_squared)$value)
}

h = function(B, tau_squared){
  return(sqrt(tau_squared)/deno_integral(B, tau_squared))
}

g = function(B, tau_squared){
  return(sqrt(tau_squared)*num_integral(B, tau_squared)/deno_integral(B, tau_squared))
}

tau_squared_optim_eqn = function(tau_squared, sparsity, B){
  return(h(B, tau_squared)+1-(1/sparsity))
}

#giving a huge upper bound 1e+100 instead of Inf as otherwise uniroot function breaks
minimax_threshold_squared = function(sparsity, B){
  return(uniroot(tau_squared_optim_eqn, c(1e-50, 1000), sparsity, B)$root)
}

#minimax threshold
minimax_threshold = function(sparsity, B){
  return(sqrt(minimax_threshold_squared(sparsity, B)))
}

#main function delivering delta PT value for given sparsity and B
predicted_delta_PT = function(sparsity, B){
  tau_squared = minimax_threshold_squared(sparsity, B)
  return((B+tau_squared+g(B, tau_squared))/(B*(1+h(B, tau_squared))))
}

minimax_threshold(0.001, 100)
