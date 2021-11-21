if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
  message(FATAL_ERROR "Algebra target directory does not exist:" ${ALGEBRA})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})

file (GLOB SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/_common/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/_common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/direct/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/direct/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/indirect/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/indirect/*.h
        )

target_sources(OSQPLIB
        PRIVATE
        ${SRC_FILES}
        )

target_include_directories(OSQPLIB
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/_common
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/direct
        ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA}/lin_sys/indirect
        )