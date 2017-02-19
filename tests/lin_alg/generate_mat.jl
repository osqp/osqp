# Open file
f = open("lin_alg/matrices.h", "w+")


# 1) Test sparse matrix construction
#-------------------------------------------------------------------------------
# Define dimensions
m = 5
n = 6
write_int(f, m, "m")
write_int(f, n, "n")

# Generate random matrix A
A = round(.6*rand(m,n)) .* randn(m,n)
write_vec_float(f, A[:], "A")

# Compute sparse CSC form
Asp = sparse(A)

# Generate sparse vectors
write_mat_sparse(f, Asp, "Asp")


# 2) Test vector operations
#-------------------------------------------------------------------------------
# Define data
t2_n = 10  # Second test vectors length
t2_v1 = randn(t2_n)
t2_v2 = randn(t2_n)
t2_sc = randn()

# Write data
write_int(f, t2_n, "t2_n")
write_vec_float(f, t2_v1, "t2_v1")
write_vec_float(f, t2_v2, "t2_v2")
write_float(f, t2_sc, "t2_sc")


# Norm of difference
t2_norm2_diff = norm(t2_v1 - t2_v2)
write_float(f, t2_norm2_diff, "t2_norm2_diff")

# Add scaled
t2_add_scaled = t2_v1 + t2_sc * t2_v2
write_vec_float(f, t2_add_scaled, "t2_add_scaled")

# Norm2 Squared
t2_norm2_sq = norm(t2_v1)^2
write_float(f, t2_norm2_sq, "t2_norm2_sq")

# Norm2
t2_norm2 = norm(t2_v1)
write_float(f, t2_norm2, "t2_norm2")

# NormInf
# t2_normInf = norm(t2_v1, Inf)
# write_float(f, t2_normInf, "t2_normInf")

# Elementwise reciprocal
t2_ew_reciprocal = 1./t2_v1
write_vec_float(f, t2_ew_reciprocal, "t2_ew_reciprocal")

# Vector product
t2_vec_prod = (t2_v1' * t2_v2)[1]
write_float(f, t2_vec_prod, "t2_vec_prod")



# 3) Matrix operations
#-------------------------------------------------------------------------------
# Define data
t3_n = 5
t3_d = randn(t3_n)
t3_A = randn(t3_n, t3_n)

# Write data
write_int(f, t3_n, "t3_n")
write_vec_float(f, t3_d, "t3_d")
write_mat_sparse(f, sparse(t3_A), "t3_A")


# Premultiply by diagonal matrix
t3_dA = diagm(t3_d)*t3_A
write_mat_sparse(f, sparse(t3_dA), "t3_dA")

# Postmultiply by diagonal matrix
t3_Ad = t3_A*diagm(t3_d)
write_mat_sparse(f, sparse(t3_Ad), "t3_Ad")

# Elementwise square
t3_A_ewsq = t3_A.^2
write_mat_sparse(f, sparse(t3_A_ewsq), "t3_A_ewsq")

# Elementwise absolute value
t3_A_ewabs = abs(t3_A)
write_mat_sparse(f, sparse(t3_A_ewabs), "t3_A_ewabs")


# 4) Matrix-vector multiplication
#-------------------------------------------------------------------------------
# Define data
m = 5
n = 4
p = 0.4

A = randn(m, n)
P = sprandn(n, n, p);
P = P + P'  # Create symmetric PSD matrix
Pu = triu(P) # Store only upper triangular part of P
x = randn(n)
y = randn(m)

# Write data
write_int(f, m, "t4_m")
write_int(f, n, "t4_n")
write_mat_sparse(f, sparse(A), "t4_A")
write_mat_sparse(f, Pu, "t4_Pu")
write_vec_float(f, x, "t4_x")
write_vec_float(f, y, "t4_y")

# Matrix-vector multiplication:  y = Ax
Ax = A*x
write_vec_float(f, Ax, "t4_Ax")

# Matrix-vector multiplication (cumulative):  y += Ax
Ax_cum = A*x + y
write_vec_float(f, Ax_cum, "t4_Ax_cum")

# Matrix-transpose-vector multiplication:  x = A'*y
ATy = A'*y
write_vec_float(f, ATy, "t4_ATy")

# Matrix-transpose-vector multiplication (cumulative):  x += A'*y
ATy_cum = A'*y + x
write_vec_float(f, ATy_cum, "t4_ATy_cum")

# Symmetric-matrix-vector multiplication (only upper triangular part is stored)
Px = P*x;
write_vec_float(f, Px, "t4_Px")

# Symmetric-matrix-vector multiplication (cummulative)
Px_cum = P*x + x;
write_vec_float(f, Px_cum, "t4_Px_cum")


# 5) Extract upper triangular matrix
#-------------------------------------------------------------------------------
# Define data
t_ut_n = 5
t_ut_A = full(sprandn(t_ut_n, t_ut_n, 0.3))
t_ut_Atriu = sparse(triu(t_ut_A))
t_ut_A = sparse(t_ut_A)

# Write data
write_int(f, t_ut_n, "t_ut_n")
write_mat_sparse(f, t_ut_A, "t_ut_A")
write_mat_sparse(f, t_ut_Atriu, "t_ut_Atriu")

# 6) Compute quadratic form using upper triangular matrix
#-------------------------------------------------------------------------------
t_qpform_n = t_ut_n
t_qpform_Atriu = t_ut_Atriu
t_qpform_A = t_qpform_Atriu + t_qpform_Atriu' - diagm(diag(t_qpform_Atriu))
t_qpform_x = randn(t_qpform_n)
t_qpform_value = (.5 * t_qpform_x' * t_qpform_A * t_qpform_x)[1]

# save data to file
# write_mat_sparse(f, t_qpform_A, "t_qpform_A")
write_mat_sparse(f, t_qpform_Atriu, "t_qpform_Atriu")
write_int(f, t_qpform_n, "t_qpform_n")
write_vec_float(f, t_qpform_x, "t_qpform_x")
write_float(f, t_qpform_value, "t_qpform_value")


# Close file
close(f)
