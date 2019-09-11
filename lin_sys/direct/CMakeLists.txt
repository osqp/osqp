# Add direct linear systems solvers

# QDLDL (Default)
# -------------------------
add_subdirectory(qdldl)


# Add other solvers if embedded option is false
if(NOT DEFINED EMBEDDED)

	# MKL Pardiso MKL
	# -----------
	# If MKL Pardiso is enabled, include pardiso directory
	if (ENABLE_MKL_PARDISO)

		# Add Pardiso interface
		add_subdirectory(pardiso)

	endif()
endif()
