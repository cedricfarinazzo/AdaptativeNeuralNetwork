FILE(GLOB_RECURSE SRCFILES *.c)

add_library(ANN ${SRCFILES})
target_link_libraries(ANN m) #math.h
target_include_directories(ANN PRIVATE ${CMAKE_SOURCE_DIR})
