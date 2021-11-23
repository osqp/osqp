include(FetchContent)

message(STATUS "Fetching/configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

FetchContent_Declare(
  qdldl
  GIT_REPOSITORY https://github.com/osqp/qdldl.git
  GIT_TAG 07ff30a3eedee4857bbbeb0778f54a86b1ea78f9
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_sources)
FetchContent_GetProperties(qdldl)

if(NOT qdldl_POPULATED)
  FetchContent_Populate(qdldl)
  # We don't actually want to build anything from here
  add_subdirectory(${qdldl_SOURCE_DIR} ${qdldl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

get_property(
  qdldl_include
  TARGET qdldlobject
  PROPERTY INCLUDE_DIRECTORIES)

if(NOT EMBEDDED)
  file(
    GLOB
    NON_EMBEDDED_SRC_FILES
    CONFIGURE_DEPENDS
    lin_sys/direct/amd/src/*.c
    lin_sys/direct/amd/include/*.h
    ../_common/csc_utils.h
    ../_common/csc_utils.c)
endif()

target_sources(
  OSQPLIB
  PRIVATE ../_common/csc_math.h
          ../_common/csc_math.c
          ../_common/kkt.h
          ../_common/kkt.c
          algebra_impl.h
          algebra_libs.c
          vector.c
          matrix.c
          ${NON_EMBEDDED_SRC_FILES}
          lin_sys/direct/qdldl_interface.h
          lin_sys/direct/qdldl_interface.c
          $<TARGET_OBJECTS:qdldlobject>)

target_include_directories(OSQPLIB PRIVATE ../_common ${CMAKE_CURRENT_SOURCE_DIR} ${qdldl_include} lin_sys/direct
                                           lin_sys/direct/amd/include)
