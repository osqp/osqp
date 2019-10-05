# Add direct linear systems solvers

# QDLDL (Default)
# -------------------------
add_subdirectory(qdldl)

# Need to add qdldlobject only here because it cannot be
# included in another object library such as linsys_qdldl
set(direct_linsys_solvers $<TARGET_OBJECTS:linsys_qdldl> $<TARGET_OBJECTS:qdldlobject>)

# NB. The second directory is added because we need to include the "qdldl_types.h" file in "qdldl_interface.h"
set(direct_linsys_solvers_includes
	"${CMAKE_CURRENT_SOURCE_DIR}/qdldl/"
	"${CMAKE_CURRENT_SOURCE_DIR}/qdldl/qdldl_sources/include")


# Add other solvers if embedded option is false
if(NOT DEFINED EMBEDDED)

# MKL Pardiso MKL
# -----------
# If MKL Pardiso is enabled, include pardiso directory
if (ENABLE_MKL_PARDISO)
	# Add Pardiso interface
	add_subdirectory(pardiso)

	add_library(linsys_pardiso OBJECT ${pardiso_sources})

	# Add parent directory for library handler
	target_include_directories(linsys_pardiso PRIVATE
														 ${pardiso_includes}
														 ${PROJECT_SOURCE_DIR}/include
														 ${PROJECT_SOURCE_DIR}/algebra/csc_tools
														 ${PROJECT_SOURCE_DIR}/lin_sys/direct/kkt_common)

	set(direct_linsys_solvers ${direct_linsys_solvers} $<TARGET_OBJECTS:linsys_pardiso>)

	set(direct_linsys_solvers_includes "${direct_linsys_solvers_includes};${CMAKE_CURRENT_SOURCE_DIR}/pardiso/")
endif()

endif()

# Add the KKT object only in normal mode and matrix-updating embedded mode (not mode 1)
if (NOT (EMBEDDED EQUAL 1))

  add_subdirectory(kkt_common)

  set(direct_linsys_solvers ${direct_linsys_solvers} $<TARGET_OBJECTS:kkt_common>)
  set(direct_linsys_solvers_includes ${direct_linsys_solvers_includes} ${kkt_common_includes})

endif()

# Pass upwards
set(direct_linsys_solvers ${direct_linsys_solvers} PARENT_SCOPE)
set(direct_linsys_solvers_includes ${direct_linsys_solvers_includes} PARENT_SCOPE)
