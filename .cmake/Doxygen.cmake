IF(NOT DOXYGEN_FOUND)
    FIND_PACKAGE(Doxygen REQUIRED dot OPTIONAL_COMPONENTS mscgen dia)
ENDIF()

if(NOT PERL_FOUND)
	find_package(Perl)
endif()


IF (DOXYGEN_FOUND AND PERL_FOUND)
    set (DOXYGEN_WORK_DIR "${CMAKE_BINARY_DIR}/doxygen")
    set (DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/doc")
    #file(MAKE_DIRECTORY "${DOXYGEN_WORK_DIR}" "${DOXYGEN_WORK_DIR}/.doc" "${DOXYGEN_WORK_DIR}/img" "${DOXYGEN_OUTPUT_DIR}")

    ADD_CUSTOM_TARGET(doc
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOXYGEN_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOXYGEN_WORK_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_WORK_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_WORK_DIR}/.doc"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_WORK_DIR}/img"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}"

        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/README.md ${DOXYGEN_WORK_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/LICENSE ${DOXYGEN_WORK_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/Doxyfile ${DOXYGEN_WORK_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src ${DOXYGEN_WORK_DIR}/src/
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc ${DOXYGEN_WORK_DIR}/doc/
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${DOXYGEN_WORK_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include ${DOXYGEN_WORK_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/.images ${DOXYGEN_WORK_DIR}/.images
        
        #  WORKING_DIRECTORY ${DOXYGEN_WORK_DIR}
        COMMAND cd "${DOXYGEN_WORK_DIR}" && ${DOXYGEN_EXECUTABLE}
        COMMAND ${CMAKE_SOURCE_DIR}/.cmake/doxy-img.sh "${DOXYGEN_WORK_DIR}/.doc/html/index.html"
    )
    
    ADD_CUSTOM_COMMAND(TARGET doc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${DOXYGEN_WORK_DIR}/.doc" ${DOXYGEN_OUTPUT_DIR}
        COMMENT "Open ./doc/html/index.html in your browser."
    )
ENDIF()
