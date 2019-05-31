configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.h"
    @ONLY
)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

FILE(GLOB_RECURSE HFILES
    RELATIVE
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    *.h
)

foreach(hf ${HFILES})
    configure_file (
        "${CMAKE_CURRENT_SOURCE_DIR}/${hf}"
        "${CMAKE_CURRENT_BINARY_DIR}/${hf}"
        COPYONLY
    )
endforeach(hf)
