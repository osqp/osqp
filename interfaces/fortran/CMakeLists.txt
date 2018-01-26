# Fortran interface compilation
enable_language(Fortran)

# Generate configure file for constants
configure_file(${PROJECT_SOURCE_DIR}/configure/glob_optsf.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/glob_optsf.h)


# Add useful libraries
set(CMAKE_FORTRAN_STANDARD_LIBRARIES "${CMAKE_FORTRAN_STANDARD_LIBRARIES} -lm -ldl")      # Include math

# Include this directory for library handler
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add shared Fortran library
add_library(osqpf SHARED 
	${CMAKE_CURRENT_SOURCE_DIR}/src/osqp_f2c.c
	${CMAKE_CURRENT_SOURCE_DIR}/src/osqpf_types.F90
	${CMAKE_CURRENT_SOURCE_DIR}/src/osqpf.F90)
target_link_libraries(osqpf osqp)
	       
# Add test executable
add_executable(osqp_demo_fortran ${CMAKE_CURRENT_SOURCE_DIR}/demo/osqp_demo_fortran.F90)
target_link_libraries(osqp_demo_fortran osqpf)
