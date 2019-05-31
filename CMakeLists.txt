cmake_minimum_required(VERSION 3.14)
project(ANN LANGUAGES C VERSION 0.1.1.4)

cmake_policy(SET CMP0079 NEW)


### CMAKE MODULE
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/.cmake" ${CMAKE_MODULE_PATH})
INCLUDE(BuildType)
INCLUDE(ConfigChecks)
INCLUDE(FindGraphviz)
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
if ((CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage") AND COVERAGE_OK)
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_COVERAGE}")
    set (CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${CMAKE_EXE_LINKER_FLAGS_COVERAGE}")
    set (CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} ${CMAKE_SHARED_LINKER_FLAGS_COVERAGE}")
endif()

set (CMAKE_C_FLAGS_RELEASE "-Ofast")
set (CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast")
set (CMAKE_MODULE_LINKER_FLAGS_RELEASE "-Ofast")

### CONFIGURE FILE

set(DEST_DIR "${CMAKE_INSTALL_PREFIX}")
set(PRIVATE_LIBS "-l${PROJECT_NAME}")
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
add_subdirectory(include/ANN/)
add_subdirectory(src/)
target_include_directories("${PROJECT_NAME}" PUBLIC
    "${PROJECT_BINARY_DIR}/include/"
)


### TEST AND MAIN
add_subdirectory(Test/)
add_subdirectory(Main/)

target_link_libraries(ANNTest ANN)
target_link_libraries(ANNMain ANN)


##### TARGETS
add_custom_target(check
    COMMAND ASAN_OPTIONS=allocator_may_return_null=1 ${CMAKE_BINARY_DIR}/Test/ANNTest -j1
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

if ((CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage") AND COVERAGE_OK)
    SETUP_TARGET_FOR_COVERAGE(
        coverage  # Name for custom target.
        ${CMAKE_BINARY_DIR}/Main/ANNMain   # Name of the test driver executable that runs the tests.
        # NOTE! This should always have a ZERO as exit code
        # otherwise the coverage generation will not complete.
        coverage  # Name of output directory.
        ANNMain  # dependency
    )
endif()


### INSTALL TARGET

# install
set(INSTALL_MAN_DIR "${CMAKE_INSTALL_PREFIX}/share/man" CACHE PATH "Installation directory for manual pages")
set(INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" CACHE PATH "Installation directory for pkgconfig (.pc) files")
set(INSTALL_LICENSE_DIR "${CMAKE_INSTALL_PREFIX}/share/licences" CACHE PATH "Installation directory for LICENSE files")

install(
    TARGETS
        ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
) # headers

install(DIRECTORY "${PROJECT_BINARY_DIR}/pkgconf/"
    DESTINATION "${INSTALL_PKGCONFIG_DIR}"
) # pkg config file

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "${INSTALL_LICENSE_DIR}/${PROJECT_NAME}/"
) # LICENSE

# uninstall
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/.cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

