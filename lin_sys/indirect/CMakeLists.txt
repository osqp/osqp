# Add indirect linear systems solvers

# Add these if embedded option is false
if(NOT DEFINED EMBEDDED)

	# MKL CG solver
	# -----------
	# If MKL Pardiso is enabled, build this too.
	# Requires its own build logic at some point.
	if (ENABLE_MKL_PARDISO)

		# Add MKL CG interface
		add_subdirectory(mkl-cg)

		#look for INTEL blas
		set(BLA_VENDOR Intel10_64lp)
		find_package(BLAS REQUIRED)

		message("-- Building with BLAS-based algebra.")
		message("-- BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

		target_link_libraries(OSQPLIB ${BLAS_LIBRARIES})

	endif()
endif()
