function write_mat_sparse(f, Asp, name)
	Asp_x = Asp.nzval
	Asp_nnz = length(Asp.nzval)
	Asp_i = Asp.rowval -1
	Asp_p = Asp.colptr -1
	write_vec_float(f, Asp_x, string(name,"_x"))
	write_int(f, Asp_nnz, string(name, "_nnz"))
	write_vec_int(f, Asp_i, string(name, "_i"))
	write_vec_int(f, Asp_p, string(name, "_p"))
end

function write_vec_float(f, x, name)
	n = size(x)[1]
	write(f, "c_float " * name)
	@printf(f, "[%d] = {", n)
	for i in 1:n
		@printf(f, "%.20f, ", x[i])
	end
	write(f, "};\n")
end

function write_vec_int(f, x, name)
	n = size(x)[1]
	write(f, "c_int " * name)
	@printf(f, "[%d] = {", n)
	for i in 1:n
		@printf(f, "%d, ", x[i])
	end
	write(f, "};\n")
end

function write_int(f, x, name)
	@printf(f, "c_int %s = %d;\n", name, x)
end

function write_float(f, x, name)
	@printf(f, "c_float %s = %.20f;\n", name, x)
end

# Reset seed
srand(10)

# Open file
f = open("matrices.h", "w+")


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


# 4) Matrix-vector operations
#-------------------------------------------------------------------------------
# Define data
t4_m = 5
t4_n = 4
t4_A = randn(t4_m, t4_n)
t4_x = randn(t4_n)
t4_y = randn(t4_m)

# Write data
write_int(f, t4_m, "t4_m")
write_int(f, t4_n, "t4_n")
write_mat_sparse(f, sparse(t4_A), "t4_A")
write_vec_float(f, t4_x, "t4_x")
write_vec_float(f, t4_y, "t4_y")

# Matrix-vector multiplication:  y = Ax
t4_Ax = t4_A*t4_x
write_vec_float(f, t4_Ax, "t4_Ax")

# Matrix-vector multiplication (cumulative):  y += Ax
t4_Ax_cum = t4_A*t4_x + t4_y
write_vec_float(f, t4_Ax_cum, "t4_Ax_cum")

# Close file
close(f)
