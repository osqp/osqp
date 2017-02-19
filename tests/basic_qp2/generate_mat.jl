# Generate Basic QP matrices
#
f = open("basic_qp2/matrices.h", "w+")

# Define Problem
P = sparse([11. 0.; 0. 0.])
q1 = [3.; 4.]
q2 = [1.; 1.]

A = sparse([-1. 0.; 0. -1.; -1. -3.; 2. 5.; 3. 4.])
lA = -1e20*ones(size(A, 1))
uA1 = [0.; 0.; -15.; 100.; 80.]
uA2 = [-2.; 0.; -20.; 100.; 80.]

n = size(P, 1)
m = size(A, 1)

# Compute solution using Convex.jl + ECOS
using Convex, ECOS

x = Variable(n)

# Problem 1
problem = minimize(0.5*quadform(x, P) + q1' * x, [A * x <= uA1])
solve!(problem, ECOSSolver())

sol_x1 = x.value
sol_lambda1 = problem.constraints[1].dual
sol_obj_value1 = problem.optval
if problem.status == :Optimal
  sol_status1 = 0  # optimal
else
  sol_status1 = 1  # infeasible
end

# Problem 2
problem = minimize(0.5*quadform(x, P) + q2' * x, [A * x <= uA2])
solve!(problem, ECOSSolver())

sol_x2 = x.value
sol_lambda2 = problem.constraints[1].dual
sol_obj_value2 = problem.optval
if problem.status == :Optimal
  sol_status2 = 0  # optimal
else
  sol_status2 = 1  # infeasible
end


# Save data
write_mat_sparse(f, P, "basic_qp2_P")
write_vec_float(f, q1, "basic_qp2_q1")
write_vec_float(f, q2, "basic_qp2_q2")

write_mat_sparse(f, A, "basic_qp2_A")
write_vec_float(f, lA, "basic_qp2_lA")
write_vec_float(f, uA1, "basic_qp2_uA1")
write_vec_float(f, uA2, "basic_qp2_uA2")

write_int(f, n, "basic_qp2_n")
write_int(f, m, "basic_qp2_m")

# Save solution: Problem 1
write_vec_float(f, sol_x1, "basic_qp2_sol_x1")
write_vec_float(f, sol_lambda1, "basic_qp2_sol_lambda1")
write_float(f, sol_obj_value1, "basic_qp2_sol_obj_value1")
write_int(f, sol_status1, "basic_qp2_sol_status1")

# Save solution: Problem 2
write_vec_float(f, sol_x2, "basic_qp2_sol_x2")
write_vec_float(f, sol_lambda2, "basic_qp2_sol_lambda2")
write_float(f, sol_obj_value2, "basic_qp2_sol_obj_value2")
write_int(f, sol_status2, "basic_qp2_sol_status2")


close(f)
