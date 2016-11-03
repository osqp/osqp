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
srand(11)



# Include files
include("lin_alg/generate_mat.jl")
include("solve_linsys/generate_mat.jl")
