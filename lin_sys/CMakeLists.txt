# Add subdirectory for each linear systems solver

if (NOT DEFINED EMBEDDED)
# Include this directory for library handler
# NB Needed for subfolders
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

endif()

# Direct solver
add_subdirectory(direct)

# Indirect solver
# TODO: Add!

# Add linsys handler if not embedded
if (NOT DEFINED EMBEDDED)

add_library(linsys_lib_handler
            ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.c)
target_include_directories(linsys_lib_handler PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS linsys_lib_handler
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()



# Combine solvers
# TODO: Add indirect ones
# Add library handler if desktop version
set(linsys_solvers
    ${direct_linsys_solvers}
    ${linsys_lib_handler}
    PARENT_SCOPE)



# Combine solvers external libraries
set(linsys_solvers_includes
    ${direct_linsys_solvers_includes}
    PARENT_SCOPE)
