# Open file
f = open("solve_linsys/matrices.h", "w+")

# 5) Linear system solve
#-------------------------------------------------------------------------------
# Define data
t5_n = 2
t5_A = [4. 1.; 1. 2.]
t5_L = [0. 0.; 0.25 0.]		# Unit diagonal is assumed !!
t5_D = [4.; 1.75]
t5_P = [0; 1]
t5_b = [1.; 2.]
t5_x = t5_A \ t5_b

# Write data
write_int(f, t5_n, "t5_n")
write_mat_sparse(f, sparse(t5_A), "t5_A")
write_mat_sparse(f, sparse(t5_L), "t5_L")
write_vec_float(f, t5_D, "t5_D")
write_vec_float(f, t5_b, "t5_b")
write_vec_float(f, t5_x, "t5_x")
write_vec_int(f, t5_P, "t5_P")


# 6) Form KKT matrix
#-------------------------------------------------------------------------------
# Define data
t6_n = 5
t6_m = 4
p = 0.3  # Control sparsity level
t6_Q = sprandn(n, n, p); t6_Q *= t6_Q  # Create symmetric matrix


# Close file
close(f)
