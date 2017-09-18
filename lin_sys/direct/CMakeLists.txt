# Add direct linear systems solvers

# Suitesparse LDL (Default)
# -------------------------
# suitesparse
add_subdirectory(suitesparse)

# suitesparse includes
include_directories(${suitesparse_ldl_includes})

# create suitesparse ldl object library
add_library(linsys_suitesparse_ldl OBJECT ${suitesparse_ldl_sources})

set(direct_linsys_solvers $<TARGET_OBJECTS:linsys_suitesparse_ldl>) 


# Pardiso MKL 
# -----------
# If MKL found, include pardiso directory
if (MKL_FOUND)
	# Add Pardiso interface
	add_subdirectory(pardiso)
	include_directories(${pardiso_includes})
	# Add actual MKL libraries and headers
	include_directories(${MKL_INCLUDE_DIR})
	set(direct_linsys_solvers_libraries ${direct_linsys_solvers_libraries} ${MKL_LIBRARIES})
	# Add Pardiso to linsys solvers
	add_library(linsys_pardiso OBJECT ${pardiso_sources})
	set(direct_linsys_solvers ${direct_linsys_solvers} $<TARGET_OBJECTS:linsys_pardiso>)
endif()



# Make direct_linsys_solvers list visible from parent directory
set(direct_linsys_solvers ${direct_linsys_solvers} PARENT_SCOPE)

# Make external linsys solvers libraries visible from parent directory
set(direct_linsys_solvers_libraries ${direct_linsys_solvers_libraries} PARENT_SCOPE)
