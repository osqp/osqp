# Create demo executable and link to the  static library
add_executable( osqp_demo osqp_demo.c )
target_link_libraries( osqp_demo osqpstatic )
