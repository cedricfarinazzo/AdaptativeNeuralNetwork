cmake_minimum_required(VERSION 3.12)
project(ANN LANGUAGES C)
set (ANN_VERSION_MAJOR 0)
set (ANN_VERSION_MINOR 1)

add_compile_options(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address  -fsanitize=leak -g3")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address  -fsanitize=leak -g3")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address  -fsanitize=leak")
set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address  -fsanitize=leak")

cmake_policy(SET CMP0079 NEW)

FILE(GLOB_RECURSE SRCFILES src/*.c)

add_library(ANN STATIC ${SRCFILES})
target_link_libraries(ANN m) #math.h
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(Test/)
add_subdirectory(Main/)

target_link_libraries(ANNTest ANN)
target_link_libraries(ANNMain ANN)


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
