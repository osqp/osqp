function render_workspace( work, output )
%RENDER_WORKSPACE Write workspace to header file.

f = fopen(output, 'w');

% Include types, constants and private header
fprintf(f, '#include \"types.h\"\n');
fprintf(f, '#include \"constants.h\"\n');
fprintf(f, '#include \"private.h\"\n\n');

% Redefine type of structure in private
fprintf(f, '// Redefine type of the structure in private\n');
fprintf(f, '// N.B. Making sure the right amount of memory is allocated\n');
fprintf(f, 'typedef struct c_priv Priv;\n\n');

% Write data structure
write_data(f, work.data);

% Write settings structure
write_settings(f, work.settings);

% Write scaling structure
write_scaling(f, work.scaling);

% Write private structure
write_private(f, work.priv);

% Define empty solution structure
write_solution(f, work.data.n, work.data.m);

% Define info structure
write_info(f);

% Define workspace structure
write_workspace(f, work.data.n, work.data.m);
    
fclose(f);

end



function write_data( f, data )
%WRITE_DATA Write data structure to file.

fprintf(f, '// Define data structure\n');

% Define matrix P
write_mat(f, data.P, 'Pdata');

% Define matrix A
write_mat(f, data.A, 'Adata');

% Define other data vectors
write_vec(f, data.q, 'qdata', 'c_float');
write_vec(f, data.l, 'ldata', 'c_float');
write_vec(f, data.u, 'udata', 'c_float');

% Define data structure
fprintf(f, 'OSQPData data = {');
fprintf(f, '%d, ', data.n);
fprintf(f, '%d, ', data.m);
fprintf(f, '&Pdata, &Adata, qdata, ldata, udata');
fprintf(f, '};\n\n');

end


function write_settings( f, settings )
%WRITE_SETTINGS Write settings structure to file.

fprintf(f, '// Define settings structure\n');
fprintf(f, 'OSQPSettings settings = {');
fprintf(f, '(c_float)%.20f, ', settings.rho);
fprintf(f, '(c_float)%.20f, ', settings.sigma);
fprintf(f, '%d, ',             settings.scaling);

fprintf(f, '%d, ',             settings.max_iter);
fprintf(f, '(c_float)%.20f, ', settings.eps_abs);
fprintf(f, '(c_float)%.20f, ', settings.eps_rel);
fprintf(f, '(c_float)%.20f, ', settings.eps_prim_inf);
fprintf(f, '(c_float)%.20f, ', settings.eps_dual_inf);
fprintf(f, '(c_float)%.20f, ', settings.alpha);

fprintf(f, '%d, ', settings.early_terminate);
fprintf(f, '%d, ', settings.early_terminate_interval);
fprintf(f, '%d',   settings.warm_start);

fprintf(f, '};\n\n');

end


function write_scaling( f, scaling )
%WRITE_SCALING Write scaling structure to file.

fprintf(f, '// Define scaling structure\n');
write_vec(f, scaling.D,    'Dscaling',    'c_float');
write_vec(f, scaling.Dinv, 'Dinvscaling', 'c_float');
write_vec(f, scaling.E,    'Escaling',    'c_float');
write_vec(f, scaling.Einv, 'Einvscaling', 'c_float');
fprintf(f, 'OSQPScaling scaling = ');
fprintf(f, '{Dscaling, Escaling, Dinvscaling, Einvscaling};\n\n');

end


function write_private( f, priv )
%WRITE_PRIVATE Write private structure to file.

fprintf(f, '// Define private structure\n');
write_mat(f, priv.L, 'priv_L')
write_vec(f, priv.Dinv, 'priv_Dinv', 'c_float')
write_vec(f, priv.P, 'priv_P', 'c_int')
fprintf(f, 'c_float priv_bp[%d];\n', length(priv.Dinv));  % Empty rhs

fprintf(f, 'Priv priv = ');
fprintf(f, '{&priv_L, priv_Dinv, priv_P, priv_bp};\n\n');

end


function write_solution( f, n, m )
%WRITE_SOLUTION Preallocate solution vectors

fprintf(f, '// Define solution\n');
fprintf(f, 'c_float xsolution[%d];\n', n);
fprintf(f, 'c_float ysolution[%d];\n\n', m);
fprintf(f, 'OSQPSolution solution = {xsolution, ysolution};\n\n');

end


function write_info( f )
%WRITE_INFO Preallocate info structure

fprintf(f, '// Define info\n');
fprintf(f, 'OSQPInfo info = {OSQP_UNSOLVED};\n\n');

end


function write_workspace( f, n, m )
%WRITE_WORKSPACE Preallocate workspace structure

fprintf(f, '// Define workspace\n');
fprintf(f, 'c_float work_x[%d];\n', n);
fprintf(f, 'c_float work_y[%d];\n', m);
fprintf(f, 'c_float work_z[%d];\n', m);
fprintf(f, 'c_float work_xz_tilde[%d];\n', n+m);
fprintf(f, 'c_float work_x_prev[%d];\n', n);
fprintf(f, 'c_float work_z_prev[%d];\n', m);
fprintf(f, 'c_float work_delta_y[%d];\n', m);
fprintf(f, 'c_float work_Atdelta_y[%d];\n', n);
fprintf(f, 'c_float work_delta_x[%d];\n', n);
fprintf(f, 'c_float work_Pdelta_x[%d];\n', n);
fprintf(f, 'c_float work_Adelta_x[%d];\n', m);
fprintf(f, 'c_float work_P_x[%d];\n', n);
fprintf(f, 'c_float work_A_x[%d];\n', m);
fprintf(f, 'c_float work_D_temp[%d];\n', n);
fprintf(f, 'c_float work_E_temp[%d];\n\n', m);

fprintf(f, 'OSQPWorkspace workspace = {\n');
fprintf(f, '&data, &priv,\n');
fprintf(f, 'work_x, work_y, work_z, work_xz_tilde,\n');
fprintf(f, 'work_x_prev, work_z_prev,\n');
fprintf(f, 'work_delta_y, work_Atdelta_y,\n');
fprintf(f, 'work_delta_x, work_Pdelta_x, work_Adelta_x,\n');
fprintf(f, 'work_P_x, work_A_x,\n');
fprintf(f, 'work_D_temp, work_E_temp,\n');
fprintf(f, '&settings, &scaling, &solution, &info};\n\n');

end


function write_vec(f, vec, name, vec_type)
%WRITE_VEC Write vector to file.

fprintf(f, '%s %s[%d] = {\n', vec_type, name, length(vec));

% Write vector elements
for i = 1 : length(vec)
    if strcmp(vec_type, 'c_float')
        fprintf(f, '(c_float)%.20f,\n', vec(i));
    else
        fprintf(f, '%i,\n', vec(i));
    end
end
fprintf(f, '};\n');

end


function write_mat(f, mat, name)
%WRITE_VEC Write matrix in CSC format to file.

write_vec(f, mat.i, [name, '_i'], 'c_int');
write_vec(f, mat.p, [name, '_p'], 'c_int');
write_vec(f, mat.x, [name, '_x'], 'c_float');

fprintf(f, 'csc %s = {', name);
fprintf(f, '%d, ', mat.nzmax);
fprintf(f, '%d, ', mat.m);
fprintf(f, '%d, ', mat.n);
fprintf(f, '%s_p, ', name);
fprintf(f, '%s_i, ', name);
fprintf(f, '%s_x, ', name);
fprintf(f, '%d};\n', mat.nz);

end