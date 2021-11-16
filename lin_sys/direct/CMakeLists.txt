# KKT COMMON TOOLS
#--------------------------
if (NOT (EMBEDDED EQUAL 1))
    add_subdirectory(kkt_common)

    set(direct_linsys_solvers $<TARGET_OBJECTS:kkt_common>)
    set(direct_linsys_solvers_includes ${kkt_common_includes})
endif()

if(ALGEBRA_DEFAULT)
    add_subdirectory(qdldl)
endif()

if(NOT DEFINED EMBEDDED)
	if (ALGEBRA_MKL)
		add_subdirectory(pardiso)
	endif()
endif()
