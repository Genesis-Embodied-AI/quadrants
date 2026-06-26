# Python, numpy, and nanobind

# nanobind requires the Development.Module component of CMake's FindPython. The component name differs on
# CMake < 3.18, hence the conditional.
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(_QD_DEV_MODULE Development)
else()
    set(_QD_DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${_QD_DEV_MODULE} REQUIRED)

# The rest of the Quadrants CMake scripts still reference the legacy uppercase FindPythonInterp/FindPythonLibs
# variables, so mirror the modern FindPython results onto them.
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
set(PYTHON_INCLUDE_DIR "${Python_INCLUDE_DIRS}")
set(PYTHON_LIBRARIES "${Python_LIBRARIES}")
set(PYTHON_VERSION_STRING "${Python_VERSION}")

execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python: Using ${Python_EXECUTABLE} as the interpreter")
message("    version: ${Python_VERSION}")
message("    include: ${Python_INCLUDE_DIRS}")
message("    numpy include: ${NUMPY_INCLUDE_DIR}")

include_directories(${NUMPY_INCLUDE_DIR})

# Detect the installed nanobind package and import its CMake config.
execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Using nanobind from: ${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)
