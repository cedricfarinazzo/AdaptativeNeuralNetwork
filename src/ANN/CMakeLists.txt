FILE(GLOB SRCFILES *.c)

add_library(ANN ${SRCFILES})
target_include_directories(ANN PRIVATE ${CMAKE_SOURCE_DIR})
