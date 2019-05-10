cmake_minimum_required(VERSION 3.12)
project(ANN LANGUAGES C VERSION 0.1.1.1)

cmake_policy(SET CMP0079 NEW)


### CMAKE MODULE
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/.cmake" ${CMAKE_MODULE_PATH})
INCLUDE(BuildType)
include(GNUInstallDirs)
include(Doxygen)
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
    INCLUDE(CodeCoverage)
endif()

### FLAGS 
add_definitions(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_C_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak -g3 -Og -DDEBUG=1")
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")
set (CMAKE_MODULE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
endif() 

set (CMAKE_C_FLAGS_RELEASE "-Ofast")
set (CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast")
set (CMAKE_MODULE_LINKER_FLAGS_RELEASE "-Ofast")

### CONFIGURE FILE
configure_file (
    "${PROJECT_SOURCE_DIR}/include/ANN/version.h.in"
    "${PROJECT_BINARY_DIR}/include/ANN/version.h"
)

set(DEST_DIR "${CMAKE_INSTALL_PREFIX}")
set(PRIVATE_LIBS "-lANN")
configure_file (
    "${PROJECT_SOURCE_DIR}/pkgconf/ANN.pc.in"
    "${PROJECT_BINARY_DIR}/pkgconf/ANN.pc"
    @ONLY
)

configure_file (
    "${PROJECT_SOURCE_DIR}/Doxyfile.in"
    "${PROJECT_BINARY_DIR}/Doxyfile"
    @ONLY
)

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
    COMMENT "Running Main on gdb !!!"
    DEPENDS ANNMain
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
    SETUP_TARGET_FOR_COVERAGE(
        coverage  # Name for custom target.
        ${CMAKE_BINARY_DIR}/Main/ANNMain   # Name of the test driver executable that runs the tests.
        # NOTE! This should always have a ZERO as exit code
        # otherwise the coverage generation will not complete.
        coverage  # Name of output directory.
        ANNMain  # dependency
    )
endif()


### INSTALL
install(
    TARGETS
      ANN
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    PATTERN "*.h.in" EXCLUDE
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(DIRECTORY "${PROJECT_BINARY_DIR}/pkgconf/" 
    DESTINATION "/usr/share/pkgconfig"
)
