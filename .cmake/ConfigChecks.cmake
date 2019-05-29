# Header checks
include(CheckIncludeFile)

check_include_file(stdlib.h            HAVE_STDLIB_H           )
check_include_file(stdio.h             HAVE_STDIO_H            )
check_include_file(fcntl.h             HAVE_FCNTL_H            )
check_include_file(math.h              HAVE_MATH_H             )
check_include_file(malloc.h            HAVE_MALLOC_H           )
check_include_file(search.h            HAVE_SEARCH_H           )
check_include_file(stat.h              HAVE_STAT_H             )
check_include_file(string.h            HAVE_STRING_H           )
check_include_file(sys/stat.h          HAVE_SYS_STAT_H         )
check_include_file(sys/time.h          HAVE_SYS_TIME_H         )
check_include_file(sys/types.h         HAVE_SYS_TYPES_H        )
check_include_file(unistd.h            HAVE_UNISTD_H           )

# Function checks
include(CheckFunctionExists)

check_function_exists(rand             HAVE_RAND           )
check_function_exists(srand            HAVE_SRAND          )
check_function_exists(printf           HAVE_PRINTF         )
check_function_exists(sprintf          HAVE_SPRINTF        )
check_function_exists(snprintf         HAVE_SNPRINTF       )
check_function_exists(vsnprintf        HAVE_VSNPRINTF      )
check_function_exists(asprintf         HAVE_ASPRINTF       )

# Type checks
# The function check_size_type also checks if the type exists
# and sets HAVE_${VARIABLE} accordingly.
include(CheckTypeSize)

check_type_size(ssize_t     SSIZE_T     )
check_type_size(size_t      SIZE_T      )

