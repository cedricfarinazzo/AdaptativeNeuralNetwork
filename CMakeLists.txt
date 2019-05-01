cmake_minimum_required(VERSION 3.12)
project(ANN LANGUAGES C)

set (ANN_VERSION_MAJOR 0)
set (ANN_VERSION_MINOR 1)

cmake_policy(SET CMP0079 NEW)


### CMAKE MODULE
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/.cmake" ${CMAKE_MODULE_PATH})
INCLUDE(BuildType)
INCLUDE(CodeCoverage)


### FLAGS 
add_definitions(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_C_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak -g3 -Og -fprofile-arcs -ftest-coverage")
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")
set (CMAKE_MODULE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")

set (CMAKE_C_FLAGS_RELEASE "-Ofast")
set (CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast")
set (CMAKE_MODULE_LINKER_FLAGS_RELEASE "-Ofast")


### STATIC LIBRARY BUILDING RULES
FILE(GLOB_RECURSE SRCFILES src/*.c)

add_library(ANN STATIC ${SRCFILES})
target_link_libraries(ANN m) #math.h
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)


### TEST AND MAIN 
add_subdirectory(Test/)
add_subdirectory(Main/)

target_link_libraries(ANNTest ANN)
target_link_libraries(ANNMain ANN)


##### TARGET
add_custom_target(check
    COMMAND ASAN_OPTIONS=allocator_may_return_null=1 ${CMAKE_BINARY_DIR}/Test/ANNTest -j0
    COMMENT "Running test !!!"
    DEPENDS ANNTest
)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/Main/ANNMain
    COMMENT "Running Main !!!"
    DEPENDS ANNMain
)

add_custom_target(gdb
    COMMAND gdb ${CMAKE_BINARY_DIR}/Main/ANNMain
    COMMENT "Running Main !!!"
    DEPENDS ANNMain
)

SETUP_TARGET_FOR_COVERAGE(
    coverage  # Name for custom target.
    ${CMAKE_BINARY_DIR}/Main/ANNMain   # Name of the test driver executable that runs the tests.
    # NOTE! This should always have a ZERO as exit code
    # otherwise the coverage generation will not complete.
    coverage  # Name of output directory.
    ANNMain  # dependency
)
