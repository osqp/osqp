# Add direct linear systems solvers

# Suitesparse LDL (Default)
# -------------------------
# suitesparse
add_subdirectory(suitesparse)

# suitesparse includes
include_directories(${suitesparse_ldl_includes})

# create suitesparse ldl object library
add_library(linsys_suitesparse_ldl OBJECT ${suitesparse_ldl_sources})

add_library(linsys_solvers $<TARGET_OBJECTS:linsys_suitesparse_ldl>) 

# Pardiso MKL 
# -----------
# If MKL found, include pardiso directory
if (MKL_FOUND)
	add_subdirectory(pardiso)
	include_directories(${pardiso_includes})
	add_library(linsys_pardiso OBJECT ${pardiso_sources})
	target_sources(linsys_solvers PRIVATE $<TARGET_OBJECTS:linsys_pardiso>)
endif()

