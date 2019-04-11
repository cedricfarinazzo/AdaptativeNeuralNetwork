FILE(GLOB SRCFILES *.c **/*.c)

add_library(ANN ${SRCFILES})
target_include_directories(ANN PRIVATE ${CMAKE_SOURCE_DIR})
