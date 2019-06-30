add_subdirectory(pyann/)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
    @ONLY
)

add_custom_target(${PYANN} ALL
    COMMAND python3 "${CMAKE_CURRENT_BINARY_DIR}/setup.py" build
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PYANN_LOW}"
)
