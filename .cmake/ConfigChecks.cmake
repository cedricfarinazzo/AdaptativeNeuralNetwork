# Header checks
include(CheckIncludeFile)

check_include_file(stdlib.h            HAVE_STDLIB_H           )
check_include_file(stdio.h             HAVE_STDIO_H            )
check_include_file(stdarg.h            HAVE_STDARG_H           )
check_include_file(unistd.h            HAVE_UNISTD_H           )
check_include_file(math.h              HAVE_MATH_H             )
check_include_file(malloc.h            HAVE_MALLOC_H           )
check_include_file(time.h              HAVE_TIME_H             )
check_include_file(sys/time.h          HAVE_SYS_TIME_H         )
check_include_file(sys/types.h         HAVE_SYS_TYPES_H        )
check_include_file(string.h            HAVE_STRING_H           )
check_include_file(graphviz/gvc.h      HAVE_GRAPHVIZ_GVC_H     )
check_include_file(graphviz/cgraph.h   HAVE_GRAPHVIZ_CGRAPH_H  )
check_include_file(graphviz/graph.h    HAVE_GRAPHVIZ_GRAPH_H   )


# Function checks
include(CheckFunctionExists)

check_function_exists(rand             HAVE_RAND           )
check_function_exists(srand            HAVE_SRAND          )
check_function_exists(rand48           HAVE_RAND48         )
check_function_exists(srand48          HAVE_SRAND48        )
check_function_exists(printf           HAVE_PRINTF         )
check_function_exists(sprintf          HAVE_SPRINTF        )
check_function_exists(snprintf         HAVE_SNPRINTF       )
check_function_exists(vsnprintf        HAVE_VSNPRINTF      )
check_function_exists(asprintf         HAVE_ASPRINTF       )
check_function_exists(vasprintf        HAVE_VASPRINTF      )

# Type checks
# The function check_size_type also checks if the type exists
# and sets HAVE_${VARIABLE} accordingly.
include(CheckTypeSize)

check_type_size(ssize_t     SSIZE_T     )
check_type_size(size_t      SIZE_T      )

