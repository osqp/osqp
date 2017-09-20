# Add direct linear systems solvers

# Suitesparse LDL (Default)
# -------------------------
# suitesparse
add_subdirectory(suitesparse)

# create suitesparse ldl object library
add_library(linsys_suitesparse_ldl OBJECT ${suitesparse_ldl_sources})

target_include_directories(linsys_suitesparse_ldl PRIVATE ${suitesparse_ldl_includes})

set(direct_linsys_solvers $<TARGET_OBJECTS:linsys_suitesparse_ldl>)

set(direct_linsys_solvers_includes ${suitesparse_ldl_includes})


# Pardiso MKL
# -----------
# If MKL found, include pardiso directory
if (MKL_FOUND)
	# Add Pardiso interface
	add_subdirectory(pardiso)
	# Define external MKL libraries to include
	set(direct_linsys_solvers_libraries ${direct_linsys_solvers_libraries} ${MKL_LIBRARIES})
	# Add Pardiso to linsys solvers
	add_library(linsys_pardiso OBJECT ${pardiso_sources})
	target_include_directories(linsys_pardiso PRIVATE  ${pardiso_includes} ${MKL_INCLUDE_DIR})
	set(direct_linsys_solvers ${direct_linsys_solvers} $<TARGET_OBJECTS:linsys_pardiso>)
	set(direct_linsys_solvers_includes ${direct_linsys_solvers_includes} ${pardiso_includes})
endif()


# Make direct_linsys_solvers list visible from parent directory
set(direct_linsys_solvers ${direct_linsys_solvers} PARENT_SCOPE)

# Make external linsys solvers libraries visible from parent directory
set(direct_linsys_solvers_libraries ${direct_linsys_solvers_libraries} PARENT_SCOPE)

# Make external linsys solvers includes visible from parent directory
set(direct_linsys_solvers_includes ${direct_linsys_solvers_includes} PARENT_SCOPE)
