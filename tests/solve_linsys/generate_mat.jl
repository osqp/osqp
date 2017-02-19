# Open file
f = open("solve_linsys/matrices.h", "w+")

# 5) Simple linear system solve
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
write_vec_float(f, 1. ./ t5_D, "t5_Dinv")
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


# 7) Random linear system solve
#-------------------------------------------------------------------------------
# Define data
#srand(1)
n = 8
In = eye(n)
P = randperm(n)-1
half_n = round(Int, n/2)
D_upp = rand(half_n) + 0.1
D_low = -rand(n-half_n) - 0.1
D = [D_upp; D_low]
L = tril(randn(n, n), -1) # We do not store unit diagonal
# A = P'LDL'P
A = In[:,P+1] * (L + In) * diagm(D) * (L' + In) * In[P+1,:]
b = randn(n)
x = A \ b

# Write data
write_int(f, n, "t7_n")
write_vec_int(f, P, "t7_P")
write_mat_sparse(f, sparse(L), "t7_L")
write_mat_sparse(f, sparse(A), "t7_A")
write_vec_float(f, 1. ./ D, "t7_Dinv")
write_vec_float(f, b, "t7_b")
write_vec_float(f, x, "t7_x")


# 8) Factorize KKT and solve a linear system
#-------------------------------------------------------------------------------
# Define data
srand(1)
n = 5
m = 6
p = 0.3  # Control sparsity level

# Create matrices A and P
A = sprandn(m, n, p)
P = sprandn(n, n, p);
P = P * P'  # Create symmetric PSD matrix
Pu = triu(P) # Store only upper triangular part of P

# Form KKT
rho = 1.6
sigma = 0.1
KKT = [P + sigma*speye(n) A'; A -1./rho*speye(m)]
rhs = randn(m+n)
x = KKT \ rhs

# Write data
write_int(f, m, "t8_m")
write_int(f, n, "t8_n")
write_float(f, rho, "t8_rho")
write_float(f, sigma, "t8_sigma")
write_mat_sparse(f, sparse(A), "t8_A")
write_mat_sparse(f, sparse(Pu), "t8_Pu")
write_vec_float(f, rhs, "t8_rhs")
write_vec_float(f, x, "t8_x")

# Close file
close(f)
