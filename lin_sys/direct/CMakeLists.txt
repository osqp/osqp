# KKT COMMON TOOLS
#--------------------------
if (NOT (EMBEDDED EQUAL 1))
    add_subdirectory(kkt_common)

    set(direct_linsys_solvers $<TARGET_OBJECTS:kkt_common>)
    set(direct_linsys_solvers_includes ${kkt_common_includes})
endif()

if(${ALGEBRA} STREQUAL "default")
    add_subdirectory(qdldl)
endif()

if(NOT DEFINED EMBEDDED)
	if (ENABLE_MKL_PARDISO)
		add_subdirectory(pardiso)
	endif()
endif()
