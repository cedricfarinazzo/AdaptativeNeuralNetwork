add_subdirectory(ANN/)
add_subdirectory(Test/)
add_subdirectory(Main/)

target_link_libraries(ANNTest ANN)
target_link_libraries(ANNMain ANN)
