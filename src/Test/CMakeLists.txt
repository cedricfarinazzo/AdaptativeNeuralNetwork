FILE(GLOB_RECURSE TSTFILES *.c)

add_executable(ANNTest ${TSTFILES})
target_link_libraries(ANNTest m) #math.h
target_link_libraries(ANNTest criterion)
