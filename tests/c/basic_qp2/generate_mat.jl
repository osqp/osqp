# Generate Basic QP matrices
#
f = open("basic_qp2/matrices.h", "w+")

# Define Problem
P = sparse([11. 0.; 0. 0.])
q = [3.; 4.]
# Ax <= uA
A = sparse([-1. 0.; 0. -1.; -1. -3.; 2. 5.; 3. 4.])
uA = [0.; 0.; -15.; 100.; 80.]

# Problem dimensions
n = size(P, 1)
m = size(A, 1)

# Define non-existing bounds
lA = -1e20*ones(size(A, 1))
lx = -1e20*ones(size(P, 1))
ux = 1e20*ones(size(P, 1))

# Save data
write_mat_sparse(f, P, "basic_qp2_P")
write_vec_float(f, q, "basic_qp2_q")

write_mat_sparse(f, A, "basic_qp2_A")
write_vec_float(f, lA, "basic_qp2_lA")
write_vec_float(f, uA, "basic_qp2_uA")

write_vec_float(f, lx, "basic_qp2_lx")
write_vec_float(f, ux, "basic_qp2_ux")

write_int(f, n, "basic_qp2_n")
write_int(f, m, "basic_qp2_m")


close(f)
