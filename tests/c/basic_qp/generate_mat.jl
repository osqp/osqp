# Generate Basic QP matrices
#
f = open("basic_qp/matrices.h", "w+")

# Define Problem
basic_qp_P = sparse([4. 1.; 1. 2.])
basic_qp_q = ones(2)

# basic_qp_A = sparse([1.0 1.0])
# basic_qp_lA = [1.0]
# basic_qp_uA = [1.0]
#
# basic_qp_lx = 0.6 * zeros(2)
# basic_qp_ux = 0.7 * ones(2)


basic_qp_A = sparse([1.0 1.0; eye(2)])
basic_qp_lA = [1.0; 0.0 * ones(2)]
basic_qp_uA = [1.0; 0.7 * ones(2)]


basic_qp_n = size(basic_qp_P, 1)
basic_qp_m = size(basic_qp_A, 1)


# Save data
write_mat_sparse(f, basic_qp_P, "basic_qp_P")
write_vec_float(f, basic_qp_q, "basic_qp_q")

write_mat_sparse(f, basic_qp_A, "basic_qp_A")
write_vec_float(f, basic_qp_lA, "basic_qp_lA")
write_vec_float(f, basic_qp_uA, "basic_qp_uA")

# write_vec_float(f, basic_qp_lx, "basic_qp_lx")
# write_vec_float(f, basic_qp_ux, "basic_qp_ux")

write_int(f, basic_qp_n, "basic_qp_n")
write_int(f, basic_qp_m, "basic_qp_m")


close(f)
