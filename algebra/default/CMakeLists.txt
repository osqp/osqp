#include(FetchContent)
#
#FetchContent_Declare(
#        qdldl
#        GIT_REPOSITORY https://github.com/osqp/qdldl.git
#        GIT_TAG 07ff30a3eedee4857bbbeb0778f54a86b1ea78f9)
#FetchContent_MakeAvailable(qdldl)
add_subdirectory(qdldl_sources)
set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

get_property(qdldl_include TARGET qdldlobject PROPERTY INCLUDE_DIRECTORIES)

if (NOT EMBEDDED)
    file (GLOB AMD_SRC_FILES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/amd/src/*.c
            ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/amd/include/*.h
            )
endif()

target_sources(OSQPLIB
        PRIVATE
        ${AMD_SRC_FILES}
        $<TARGET_OBJECTS:qdldlobject>
        )

target_include_directories(OSQPLIB
        PRIVATE
        ${qdldl_include}
        lin_sys/direct/amd/include
        )