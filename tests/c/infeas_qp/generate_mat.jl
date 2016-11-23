# Generate Basic QP matrices
#
f = open("infeas_qp/matrices.h", "w+")

# Define Problem
P = spzeros(2, 2)
q = ones(2)

A = sparse([eye(2); 1.0 1.0])
lA = [0.0; 0.0; -1.0]
uA = [1e20; 1e20; -1.0]

m = size(basic_qp_A, 1)
n = size(basic_qp_A, 2)

# Save data
write_mat_sparse(f, P, "infeas_qp_P")
write_vec_float(f, q, "infeas_qp_q")

write_mat_sparse(f, A, "infeas_qp_A")
write_vec_float(f, lA, "infeas_qp_lA")
write_vec_float(f, uA, "infeas_qp_uA")

write_int(f, n, "infeas_qp_n")
write_int(f, m, "infeas_qp_m")


close(f)
