FILE(GLOB_RECURSE SRCFILES *.c)

include_directories(${GRAPHVIZ_INCLUDE_DIR})
add_library(${PROJECT_NAME} STATIC ${SRCFILES})
target_link_libraries(${PROJECT_NAME} 
    m #math.h
    ${GRAPHVIZ_GVC_LIBRARY} #graphviz
    ${GRAPHVIZ_CGRAPH_LIBRARY}
    ${GRAPHVIZ_CDT_LIBRARY}
)
