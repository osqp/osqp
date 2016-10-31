# function write_mat(f, A, name)
# 	m, n = size(A)
# 	write(f, "c_float " * name)
# 	@printf(f, "[%d][%d] = {\n", m, n)
# 	for i in 1:m
# 		write(f, "  {")
# 		for j in 1:n
# 			@printf(f, "%f, ", A[i,j])
# 		end
# 		write(f, "},\n")
# 	end
# 	write(f, "};\n\n")
# end


function write_vec_float(f, x, name)
	n = size(x)[1]
	write(f, "c_float " * name)
	@printf(f, "[%d] = {", n)
	for i in 1:n
		@printf(f, "%f, ", x[i])
	end
	write(f, "};\n")
end

function write_vec_int(f, x, name)
	n = size(x)[1]
	write(f, "c_int " * name)
	@printf(f, "[%d] = {", n)
	for i in 1:n
		@printf(f, "%f, ", x[i])
	end
	write(f, "};\n")
end

function write_int(f, x, name)
	@printf(f, "c_int %s = %d;\n", name, x)
end

function write_float(f, x, name)
	@printf(f, "c_float %s = %f;\n", name, x)
end

# Reset seed
srand(3)

# Open file
f = open("matrices.h", "w+")

# 1) Test sparse matrix construction
#-------------------------------------------------------------------------------
# Define dimensions
m = 5
n = 6
write_int(f, m, "m")
write_int(f, n, "n")
# @printf(f, "c_int m = %d;\n", m)
# @printf(f, "c_int n = %d;\n", n)

# Generate random matrix A
A = round(.6*rand(m,n)) .* randn(m,n)
write_vec_float(f, A[:], "A")

# Compute sparse CSC form
Asp = sparse(A)

# Generate sparse vectors
Asp_x = Asp.nzval
Asp_nnz = length(Asp.nzval)
Asp_i = Asp.rowval -1
Asp_p = Asp.colptr -1
write_vec_float(f, Asp_x, "Asp_x")
write_int(f, Asp_nnz, "Asp_nnz")
# @printf(f, "c_int Asp_nnz = %d;\n", Asp_nnz)
write_vec_int(f, Asp_i, "Asp_i")
write_vec_int(f, Asp_p, "Asp_p")

# 2) Test vector norms
#-------------------------------------------------------------------------------
# Define data
t2_n = 10  # Second test vectors length
t2_v1 = randn(t2_n)
t2_v2 = randn(t2_n)
t2_sc = randn()
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
t2_normInf = norm(t2_v1, Inf)
write_float(f, t2_normInf, "t2_normInf")





# Close file
close(f)


# B = round(.6*rand(m,n)) .* randn(m,n)
# P = round(.6*rand(n,n)) .* randn(n,n)
# P = P*P' + .1*eye(n)
# x = randn(n)
# y = randn(m)
# rho = 5.

# write_vec(f, A[:], "A")
# write_mat(f, A', "AT")
# write_mat(f, B, "B")
# write_mat(f, P, "P")
# write_vec(f, x, "x")
# write_vec(f, y, "y")
# write_vec(f, A*x, "Ax")
# write_vec(f, A'*y, "ATy")
# write_mat(f, A*diagm(x), "AE")
# write_mat(f, diagm(y)*A, "FA")
# write_mat(f, [A;B], "AvcatB")
# write_mat(f, [A B], "AhcatB")
# write_mat(f, P + rho*eye(n), "Prho")
# write_vec(f, P\x, "Pinvx")
# write_mat(f, P + rho*eye(n), "PrhoI")
# write_mat(f, rho*eye(n), "rhoI")

# @printf(f, "double rho = %f;\n", rho)
# close(f)
