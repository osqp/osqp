# Add the OSQP headers
target_include_directories( OSQPLIB
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/public
)
