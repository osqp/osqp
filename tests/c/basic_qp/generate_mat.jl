# Generate Basic QP matrices
#
f = open("basic_qp/matrices.h", "w+")

# Define Problem
P = sparse([4.  1.; 1. 2.])
q = ones(2)

A = sparse([1.0 1.0; eye(2)])
lA = [1.0; 0.0 * ones(2)]
uA = [1.0; 0.7 * ones(2)]

n = size(P, 1)
m = size(A, 1)

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
else
  sol_status = 1  # infeasible
end

# Dummy data
q_new = [2.5; 3.2]
lA_new = [0.8; -3.4; -11.]
uA_new = [1.6; 1.0; 15.]

# Save data
write_mat_sparse(f, P, "basic_qp_P")
write_vec_float(f, q, "basic_qp_q")

write_mat_sparse(f, A, "basic_qp_A")
write_vec_float(f, lA, "basic_qp_lA")
write_vec_float(f, uA, "basic_qp_uA")

write_int(f, n, "basic_qp_n")
write_int(f, m, "basic_qp_m")

# Save solution
write_vec_float(f, sol_x, "basic_qp_sol_x")
write_vec_float(f, sol_lambda, "basic_qp_sol_lambda")
write_float(f, sol_obj_value, "basic_qp_sol_obj_value")
write_int(f, sol_status, "basic_qp_sol_status")

# Save dummy data
write_vec_float(f, q_new, "basic_qp_q_new")
write_vec_float(f, lA_new, "basic_qp_lA_new")
write_vec_float(f, uA_new, "basic_qp_uA_new")

close(f)
