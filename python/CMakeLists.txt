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

add_custom_target("${PYANN}_install"
    COMMAND python3 "${CMAKE_CURRENT_BINARY_DIR}/setup.py" install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PYANN}"
)

add_custom_target("${PYANN}_sdist"
    COMMAND python3 "${CMAKE_CURRENT_BINARY_DIR}/setup.py" sdist
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PYANN}"
)

add_custom_target("${PYANN}_bdist"
    COMMAND python3 "${CMAKE_CURRENT_BINARY_DIR}/setup.py" bdist
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PYANN}"
)
