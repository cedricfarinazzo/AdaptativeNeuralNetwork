# Find if a Python module is installed
function(find_python_module module)
	string(TOUPPER ${module} module_upper)
	if(NOT PY_${module_upper})
		# A module's location is usually a directory, but for binary modules
		# it's a .so file.
		execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" 
			"import ${module}"
			OUTPUT_VARIABLE ${module}_not_found
			ERROR_QUIET 
                        )
	endif(NOT PY_${module_upper})
        if (NOT ${module}_not_found)
            set(PY_${module_upper}_FOUND TRUE CACHE STRING "${module} python module")
            MESSAGE(STATUS "Python3 ${module} found")
        else()
            set(PY_${module_upper}_FOUND FALSE CACHE)
            MESSAGE(STATUS "Python3 ${module} NOT FOUND")
        endif()
endfunction(find_python_module)
