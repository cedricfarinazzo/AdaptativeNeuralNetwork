cmake_minimum_required(VERSION 3.12)
project(ANN LANGUAGES C)
set (ANN_VERSION_MAJOR 0)
set (ANN_VERSION_MINOR 1)

add_compile_options(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_C_FLAGS_DEBUG "-fsanitize=address -g3")

cmake_policy(SET CMP0079 NEW)

#set (CMAKE_SHARED_LINKER_FLAGS "-lcriterion -fprofile-arcs –ftest-coverage")

add_subdirectory(src/)

add_custom_target(check
    COMMAND ${CMAKE_BINARY_DIR}/src/Test/ANNTest
    COMMENT "Running test !!!"
    DEPENDS ANNTest
)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/src/Main/ANNMain
    COMMENT "Running Main !!!"
    DEPENDS ANNMain
)
