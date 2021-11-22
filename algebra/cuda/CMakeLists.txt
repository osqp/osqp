file (GLOB SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
        )

if (IS_WINDOWS)
    set(HELPER_CUDA_H $ENV{NVCUDASAMPLES_ROOT}/common/inc)
else ()
    set(HELPER_CUDA_H ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif()

target_sources(OSQPLIB
        PUBLIC
        ${SRC_FILES}
        )

target_include_directories(OSQPLIB
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HELPER_CUDA_H}
        )