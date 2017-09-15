# Add direct linear systems solvers

# suitesparse
add_subdirectory(suitesparse)

# suitesparse includes
include_directories(${suitesparse_ldl_includes})

# include pardiso if there is MKL
# pardiso
add_subdirectory(pardiso)


# create library
add_library(linsys_suitesparse_ldl OBJECT ${suitesparse_ldl_sources})
