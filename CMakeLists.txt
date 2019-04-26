cmake_minimum_required(VERSION 3.12)
project(ANN LANGUAGES C)
set (ANN_VERSION_MAJOR 0)
set (ANN_VERSION_MINOR 1)

add_compile_options(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address  -fsanitize=leak -g")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address  -fsanitize=leak -g")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address  -fsanitize=leak")
set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address  -fsanitize=leak")

cmake_policy(SET CMP0079 NEW)


add_subdirectory(src/)

add_custom_target(check
    COMMAND ${CMAKE_BINARY_DIR}/src/Test/ANNTest
    COMMENT "Running test !!!"
    DEPENDS ANNTest
)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/src/Main/ANNMain -j0
    COMMENT "Running Main !!!"
    DEPENDS ANNMain
)
