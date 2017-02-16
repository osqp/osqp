# Minimum version required
cmake_minimum_required (VERSION 3.5)

# Project name
project (osqp)

# project version --> Does it make sense to have it here?
# set(VERSION_MAJOR 0)
# set(VERSION_MINOR 0)
# set(VERSION_PATCH 0)

# Set the output folder where your program will be created
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)



# Set options
# ----------------------------------------------------------------------------
# Is printing enabled?
option (PRINTING "Enable solver printing" ON)
# Is printing enabled?
option (PROFILING "Enable solver profiling (timing)" ON)
# Use floats instead of integers
option (FLOAT "Use float numbers instead of doubles" OFF)
# Use long integers for indexing
option (LONG "Use long integers for indexing" ON)


# Apply options
# ----------------------------------------------------------------------------
if (PRINTING)
    add_definitions(-DPRINTING)
endif (PRINTING)
if (PROFILING)
    add_definitions(-DPROFILING)
endif (PROFILING)
if (FLOAT)
    add_definitions(-DDFLOAT)
endif (FLOAT)
if (LONG)
    add_definitions(-DDLONG)
endif (LONG)


# Detect operating system
# ---------------------------------------------------------------------------
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_definitions(-DIS_LINUX)
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_definitions(-DIS_MAC)
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_definitions(-DIS_WINDOWS)
endif()



# Set Compiler flags
# ---------------------------------------------------------------------------
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -lm")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -g -lm")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD_LIBRARIES m)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CMAKE_C_STANDARD_LIBRARIES ${CMAKE_C_STANDARD_LIBRARIES} rt)
endif()


# Include header directory
# ---------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(
    osqp_src
    auxil.c
    cs.c
    kkt.c
    lin_alg.c
    osqp.c
    polish.c
    proj.c
    scaling.c
    util.c)

set(
    osqp_headers
    auxil.h
    constants.h
    cs.h
    glob_opts.h
    kkt.h
    lin_alg.h
    lin_sys.h
    osqp.h
    polish.h
    proj.h
    scaling.h
    util.h
)

# Add linear systems solver
add_subdirectory(linsys)

# add osqp libraries static and shared
add_library (libosqpdir SHARED ${osqp_src} ${osqp_headers})
target_link_libraries (libosqpdir linsys_direct)

add_library (libosqpdir STATIC ${osqp_src} ${osqp_headers})
target_link_libraries (libosqpdir linsys_direct)

# add the executable
add_executable (osqp_demo_direct ${PROJECT_SOURCE_DIR}/examples/osqp_demo_direct.cpp)
target_link_libraries (osqp_demo_direct libosqpdir)



# Add testing
# TODO:Complete!
# include(CTest)
# enable_testing()
# add_test(osqp_tester_direct ${PROJECT_BINARY_DIR}/osqp_tester_direct)
