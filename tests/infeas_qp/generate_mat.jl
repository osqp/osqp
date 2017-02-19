# Generate Basic QP matrices
#
f = open("infeas_qp/matrices.h", "w+")

# Define Problem
P = spzeros(2, 2)
q = ones(2)

A = sparse([eye(2); 1.0 1.0])
lA = [0.0; 0.0; -1.0]
uA = [100; 100; -1.0]

m = size(A, 1)
n = size(A, 2)

# Compute solution using Convex.jl + ECOS
using Convex, ECOS

x = Variable(n)
problem = minimize(0.5*quadform(x, P) + q' * x, [A * x >= lA, A * x <= uA])
solve!(problem, ECOSSolver())

sol_x = x.value
sol_lambda = problem.constraints[2].dual - problem.constraints[1].dual
sol_obj_value = problem.optval
if problem.status == :Optimal
  sol_status = 0  # optimal
elseif problem.status == :Infeasible
  sol_status = 1  # infeasible
else
  sol_status = -1  # other
end


# Save data
write_mat_sparse(f, P, "infeas_qp_P")
write_vec_float(f, q, "infeas_qp_q")

write_mat_sparse(f, A, "infeas_qp_A")
write_vec_float(f, lA, "infeas_qp_lA")
write_vec_float(f, uA, "infeas_qp_uA")

write_int(f, n, "infeas_qp_n")
write_int(f, m, "infeas_qp_m")

# Save solution
write_vec_float(f, sol_x, "infeas_qp_sol_x")
write_vec_float(f, sol_lambda, "infeas_qp_sol_lambda")
write_float(f, sol_obj_value, "infeas_qp_sol_obj_value")
write_int(f, sol_status, "infeas_qp_sol_status")


close(f)
