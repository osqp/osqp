# Minimum version required
cmake_minimum_required (VERSION 3.2)





# Project name
project (osqp)

# Set the output folder where your program will be created
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

# Some non-standard CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/configure/cmake/")
INCLUDE(FindPythonModule)
INCLUDE(Utils)


# Detect operating system
# ----------------------------------------------
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(IS_LINUX ON)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(IS_MAC ON)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(IS_WINDOWS ON)
endif()



# Set options
# ----------------------------------------------

# Are unittests generated?
option (UNITTESTS "Enable unittests generation" ON)

# Is the code generated for embedded platforms?
#   1 :   Yes. Matrix update not allowed.
#   2 :   Yes. Matrix update allowed.
if (NOT DEFINED EMBEDDED)
    message(STATUS "Embedded is OFF")
else()
    message(STATUS "Embedded is ${EMBEDDED}")
endif()

# Is printing enabled?
option (PRINTING "Enable solver printing" ON)
if (DEFINED EMBEDDED)
    message(STATUS "Disabling printing for embedded")
    set(PRINTING OFF)
endif()
message(STATUS "Printing is ${PRINTING}")


# Is profiling enabled?
option (PROFILING "Enable solver profiling (timing)" ON)
if (DEFINED EMBEDDED)
    message(STATUS "Disabling profiling for embedded")
    set(PROFILING OFF)
endif()
message(STATUS "Profiling is ${PROFILING}")

# Is user interrupt enabled?
option (CTRLC "Enable user interrupt (Ctrl-C)" ON)
if (DEFINED EMBEDDED)
    message(STATUS "Disabling user interrupt for embedded")
    set(CTRLC OFF)
endif()
message(STATUS "User interrupt is ${CTRLC}")

# Use floats instead of integers
option (DFLOAT "Use float numbers instead of doubles" OFF)
message(STATUS "Floats are ${DFLOAT}")

# Use long integers for indexing
option (DLONG "Use long integers for indexing" ON)
message(STATUS "Long integers are ${DLONG}")

# Add code coverage
option (COVERAGE "Perform code coverage" OFF)
message(STATUS "Code coverage is ${DLONG}")


# Generate header file with the global options
# ---------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure/glob_opts.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/glob_opts.h)


# Set Compiler flags
# ----------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

if (NOT MSVC)

    if (COVERAGE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm")      # Include math
    # Include real time library in linux
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lrt")
    endif()
endif (NOT MSVC)


# Include header directory
# ----------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# Set sources
# ----------------------------------------------
if(NOT DEFINED EMBEDDED)
    set(
        osqp_src
        src/auxil.c
        src/cs.c
        src/ctrlc.c
        src/kkt.c
        src/lin_alg.c
        src/lin_sys.c
        src/osqp.c
        src/polish.c
        src/proj.c
        src/scaling.c
        src/util.c
    )

    set(
        osqp_headers
        include/auxil.h
        include/constants.h
        include/cs.h
        include/ctrlc.h
        include/glob_opts.h
        include/kkt.h
        include/lin_alg.h
        include/lin_sys.h
        include/osqp.h
        include/types.h
        include/polish.h
        include/proj.h
        include/scaling.h
        include/util.h
    )
else()
    # If embedded is enabled, do not use cs, polish
    set(
        osqp_src
        src/auxil.c
        src/lin_alg.c
        src/lin_sys.c
        src/osqp.c
        src/kkt.c
        src/proj.c
        src/scaling.c
        src/util.c
    )

    set(
        osqp_headers
        include/auxil.h
        include/constants.h
        include/glob_opts.h
        include/lin_alg.h
        include/lin_sys.h
        include/kkt.h
        include/osqp.h
        include/types.h
        include/proj.h
        include/scaling.h
        include/util.h
    )
endif()



# if we are building the Python interface, let's look for Python
# and set some options
# -----------------------------------------------------------------
if (PYTHON)

    # Look for Python libraries
    find_package(PythonInterp ${PYTHON_VER_NUM})
    find_package(PythonLibs ${PYTHON_VER_NUM})

    if (NOT PYTHONLIBS_FOUND)
            message( FATAL_ERROR "You need Python libraries to build the Python interface" )
    endif (NOT PYTHONLIBS_FOUND)

    # Include directories for Python headers
    include_directories(${PYTHON_INCLUDE_DIRS})

    # Pass PYTHON flag to C compiler
    add_definitions(-DPYTHON)

    if (UNITTESTS)
        # Disable unittests
        message(STATUS "Disabling UNITTESTS because we are building Python interface")
        set(UNITTESTS OFF)
    endif (UNITTESTS)

endif (PYTHON)


# if we are building the Matlab interface, let's look for Matlab
# and set some options
# -----------------------------------------------------------------
if (MATLAB)
    find_package(Matlab)
    if (NOT Matlab_FOUND)
        message( FATAL_ERROR "You need Matlab libraries to build the Matlab interface" )
    endif (NOT Matlab_FOUND)

    # Include directories for Python headers
    include_directories(${Matlab_INCLUDE_DIRS})

    # Pass MATLAB flag to C compiler
    add_definitions(-DMATLAB)

    if (UNITTESTS)
        # Disable unittests
        message(STATUS "Disabling UNITTESTS because we are building Matlab interface")
        set(UNITTESTS OFF)
    endif (UNITTESTS)

endif (MATLAB)



# Create  Library
# ----------------------------------------------

# Add linear systems solver
add_subdirectory(lin_sys)

# Add library
# N.B. Add all the linear system solvers here
add_library (osqpstatic STATIC ${osqp_src} ${osqp_headers} $<TARGET_OBJECTS:linsys_suitesparse_ldl>)

if (MATLAB)
target_link_libraries (osqpstatic ${Matlab_LIBRARIES})
endif (MATLAB)

if (PYTHON)
target_link_libraries (osqpstatic ${PYTHON_LIBRARIES})
endif (PYTHON)

# If we are building Python or Matlab interface:
#   - do not build shared library
#   - do not build demo
if (NOT PYTHON AND NOT MATLAB)
    # Create osqp shared library
    # N.B. Add all the linear system solvers here
    add_library (osqp SHARED ${osqp_src} ${osqp_headers} $<TARGET_OBJECTS:linsys_suitesparse_ldl>)

    # Create demo executable (linked to static library)
    add_executable (osqp_demo ${PROJECT_SOURCE_DIR}/examples/osqp_demo.c)
    target_link_libraries (osqp_demo osqpstatic)
endif (NOT PYTHON AND NOT MATLAB)


# Add testing
# ----------------------------------------------
# Add custom command to generate tests
if (UNITTESTS)
    find_package(PythonInterp)
    if(NOT PYTHONINTERP_FOUND)
        message( FATAL_ERROR "You need python installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
    endif()

    INCLUDE(FindPythonModule)
    find_python_module(numpy)
    IF(NOT NUMPY_FOUND)
        message( FATAL_ERROR "You need numpy python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
    ENDIF()

    find_python_module(scipy)
    # Check scipy version for sparse.random functionalities
    IF((NOT SCIPY_FOUND) OR (SCIPY_VERSION VERSION_LESS 0.17.0))
        message( FATAL_ERROR "You need scipy python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
    ENDIF()

    find_python_module(__future__)
    IF(NOT __FUTURE___FOUND)
        message( FATAL_ERROR "You need future python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
    ENDIF()

    # Add test_headers and codegen_test_headers
    add_subdirectory(tests)

    # Generating tests.stamp so that the test data are not always generated
    # set(data_timestamp ${PROJECT_SOURCE_DIR}/tests/tests_data.stamp)
    add_custom_command(
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
        COMMAND ${PYTHON_EXECUTABLE} generate_tests_data.py
        DEPENDS ${PROJECT_SOURCE_DIR}/tests/generate_tests_data.py
        OUTPUT ${codegen_test_headers}
        COMMENT "Generating unittests data files using Python"
    )


    # Direct linear solver testing
    include_directories(tests)
    add_executable(osqp_tester
                ${PROJECT_SOURCE_DIR}/tests/osqp_tester.c ${PROJECT_SOURCE_DIR}/tests/minunit.h
                ${test_headers}
                ${codegen_test_headers})
    target_link_libraries (osqp_tester osqpstatic)

    # Add testing
    include(CTest)
    enable_testing()
    add_test(NAME tester COMMAND $<TARGET_FILE:osqp_tester>)
endif()
