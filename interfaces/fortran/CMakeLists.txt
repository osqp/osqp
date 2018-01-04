# Fortran interface compilation
enable_language(Fortran)

# Add useful libraries
set(CMAKE_FORTRAN_STANDARD_LIBRARIES "${CMAKE_FORTRAN_STANDARD_LIBRARIES} -lm -ldl")      # Include math

# Include this directory for library handler
# NB Needed for subfolders
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add shared Fortran library
add_library(osqpf SHARED 
	    osqp_f2c.c
	    osqpf_types.F90
	    osqpf.F90)
target_link_libraries(osqpf osqp)
	       
# Add test executable
add_executable(osqp_tester_fortran test_osqpf.F90)
target_link_libraries(osqp_tester_fortran osqpf)
