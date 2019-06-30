FILE(GLOB_RECURSE SWIGFILES *.i)

swig_add_library(
    "${PYANN_LOW}" 
    LANGUAGE python
    SOURCES "${SWIGFILES}"
)
swig_link_libraries(
    "${PYANN_LOW}"
    ANN
    ${PYTHON_LIBRARIES}
)
set(PYANN_PATH_LOW "${CMAKE_CURRENT_BINARY_DIR}/_${PYANN_LOW}.so" PARENT_SCOPE)

FILE(GLOB_RECURSE PYFILES
    RELATIVE
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    *.py
)

foreach(f ${PYFILES})
    configure_file (
        "${CMAKE_CURRENT_SOURCE_DIR}/${f}"
        "${CMAKE_CURRENT_BINARY_DIR}/${f}"
        COPYONLY
    )
endforeach(f)
