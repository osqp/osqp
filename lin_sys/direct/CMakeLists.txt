# Add direct linear systems solvers

# QDLDL (Default)
# -------------------------
add_subdirectory(qdldl)

# Need to add qdldlobject only here because it cannot be
# included in another object library such as linsys_qdldl
set(direct_linsys_solvers linsys_qdldl)

set(direct_linsys_solvers_includes "${CMAKE_CURRENT_SOURCE_DIR}/qdldl/")


# Add other solvers if embedded option is false
if(NOT DEFINED EMBEDDED)

# MKL Pardiso MKL
# -----------
# If MKL Pardiso is enabled, include pardiso directory
if (ENABLE_MKL_PARDISO)
	# Add Pardiso interface
	add_subdirectory(pardiso)

	add_library(linsys_pardiso ${pardiso_sources})

	# Add parent directory for library handler
	target_include_directories(linsys_pardiso PRIVATE  ${pardiso_includes} ${PROJECT_SOURCE_DIR}/include)

	set(direct_linsys_solvers ${direct_linsys_solvers} linsys_pardiso)

	set(direct_linsys_solvers_includes "${direct_linsys_solvers_includes};${CMAKE_CURRENT_SOURCE_DIR}/pardiso/")
endif()


endif()

# Make direct_linsys_solvers list visible from parent directory
set(direct_linsys_solvers ${direct_linsys_solvers} PARENT_SCOPE)

# Make external linsys solvers includes visible from parent directory
set(direct_linsys_solvers_includes ${direct_linsys_solvers_includes} PARENT_SCOPE)
