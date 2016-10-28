function write_mat(f, A, name)
	m, n = size(A)
	write(f, "double " * name)
	@printf(f, "[%d][%d] = {\n", m, n)
	for i in 1:m
		write(f, "  {")
		for j in 1:n
			@printf(f, "%f, ", A[i,j])
		end
		write(f, "},\n")
	end
	write(f, "};\n\n")
end


function write_vec(f, x, name)
	n = size(x)[1]
	write(f, "double " * name)
	@printf(f, "[%d] = {", n)
	for i in 1:n
		@printf(f, "%f, ", x[i])
	end
	write(f, "};\n\n")
end


m = 5
n = 20
srand(0)
A = round(.6*rand(m,n)) .* randn(m,n)
B = round(.6*rand(m,n)) .* randn(m,n)
P = round(.6*rand(n,n)) .* randn(n,n)
P = P*P' + .1*eye(n)
x = randn(n)
y = randn(m)
rho = 5.

f = open("matrices.h", "w+")
write_mat(f, A, "A")
write_mat(f, A', "AT")
write_mat(f, B, "B")
write_mat(f, P, "P")
write_vec(f, x, "x")
write_vec(f, y, "y")
write_vec(f, A*x, "Ax")
write_vec(f, A'*y, "ATy")
write_mat(f, A*diagm(x), "AE")
write_mat(f, diagm(y)*A, "FA")
write_mat(f, [A;B], "AvcatB")
write_mat(f, [A B], "AhcatB")
write_mat(f, P + rho*eye(n), "Prho")
write_vec(f, P\x, "Pinvx")
write_mat(f, P + rho*eye(n), "PrhoI")
write_mat(f, rho*eye(n), "rhoI")
@printf(f, "int m = %d;\n", m)
@printf(f, "int n = %d;\n", n)
@printf(f, "double rho = %f;\n", rho)
close(f)
