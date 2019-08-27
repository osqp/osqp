
set(kkt_common_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.h
)

set(kkt_common_src
    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.c
)

# Create object library for kkt solvers
add_library(kkt_common OBJECT ${kkt_common_src})

target_include_directories(kkt_common PRIVATE ${kkt_common_includes}
                           ${PROJECT_SOURCE_DIR}/algebra/csc_tools
                           ${PROJECT_SOURCE_DIR}/include
                           )
