if (DLONG)
    set(MKL_INTERFACE "ilp64")
else()
    set(MKL_INTERFACE "lp64")
endif()

find_package(MKL CONFIG REQUIRED)

target_sources(OSQPLIB
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/matrix.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/vector.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/algebra_libs.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h"
)


target_include_directories(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(OSQPLIB PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(OSQPLIB PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_link_libraries(OSQPLIB PUBLIC $<LINK_ONLY:MKL::MKL>)