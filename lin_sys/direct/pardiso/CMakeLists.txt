# Set source files
target_sources(OSQPLIB
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/pardiso_interface.h
      ${CMAKE_CURRENT_SOURCE_DIR}/pardiso_interface.c
      ${CMAKE_CURRENT_SOURCE_DIR}/pardiso_loader.h
      ${CMAKE_CURRENT_SOURCE_DIR}/pardiso_loader.c
)
