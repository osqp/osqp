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
t6_p = 0.3  # Control sparsity level
t6_rho = 1.6
t6_P = sprandn(t6_n, t6_n, t6_p); t6_P = (t6_P + t6_P')/2.0  # Create symmetric matrix
t6_P = triu(t6_P)# Store only upper triangular part of P

# Compute KKT
t6_PrhoI = t6_P + t6_rho*speye(size(t6_P,1))
t6_A = sprandn(t6_m, t6_n, t6_p)

# KKT (only one part)
# t6_KKT = [t6_PrhoI            spzeros(size(t6_A')...);
#           spzeros(size(t6_A)...)     spzeros(size(t6_A,1), size(t6_A,1))]
# Complete KKT
t6_KKT = [t6_PrhoI            t6_A';
          spzeros(size(t6_A)...)     -1./t6_rho*speye(size(t6_A,1))]


# Save data and matrices  to file
write_float(f, t6_rho, "t6_rho")
write_int(f, t6_n, "t6_n")
write_int(f, t6_m, "t6_m")

write_mat_sparse(f, t6_P, "t6_P")
write_mat_sparse(f, t6_PrhoI, "t6_PrhoI")
write_mat_sparse(f, t6_A, "t6_A")
write_mat_sparse(f, t6_KKT, "t6_KKT")


# Close file
close(f)
