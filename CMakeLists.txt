cmake_minimum_required(VERSION 3.31) # Minimum required CMake version

project(autodiff_project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set for OpenMp
find_package(OpenMP REQUIRED)

# Define include directories for the entire project
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src # Add src as an include just in case, though headers should be in include
)

# --- Build the core autodiff library ---
# This library will contain the forward and reverse autodiff implementations.
# It's defined as an INTERFACE library because the implementation is mostly header-only,
# but this still allows us to manage include directories and dependencies clearly.
# If you start adding .cpp files to forward/, change this to add_library(autodiff STATIC ...)
add_library(autodiff INTERFACE)

# Link the necessary include directories to the autodiff library
target_include_directories(autodiff INTERFACE
        ${CMAKE_SOURCE_DIR}/include/autodiff/forward
        ${CMAKE_SOURCE_DIR}/include/autodiff/reverse

)

# If src/autodiff/reverse has .cpp files that need compiling into this library:
# set(AUTODIFF_REVERSE_SOURCES
#     src/autodiff/reverse/EigenSupport.hpp # Headers should generally not be listed as sources unless they are compiled units (rare)
#     src/autodiff/reverse/Functions.hpp
#     src/autodiff/reverse/Node.hpp
#     src/autodiff/reverse/NodeManager.hpp
#     src/autodiff/reverse/Tape.hpp
#     src/autodiff/reverse/test.cpp # This looks like a test file, likely should be in a separate executable
#     src/autodiff/reverse/utils.hpp
#     src/autodiff/reverse/Var.hpp
#     # Add any actual .cpp source files here
# )
# target_sources(autodiff PRIVATE ${AUTODIFF_REVERSE_SOURCES})
# Change add_library(autodiff INTERFACE) to add_library(autodiff STATIC ...) if you add sources.

# --- Build a library for models and optimizers ---
# This groups headers and potentially sources for models and optimizers.
add_library(ml_components INTERFACE)

# Link their include directories
target_include_directories(ml_components INTERFACE
        ${CMAKE_SOURCE_DIR}/include/models
        ${CMAKE_SOURCE_DIR}/include/optimizer
        ${CMAKE_SOURCE_DIR}/include/newton # Include newton headers too
)

# Add any source files for models/optimizers if they exist outside includes
set(ML_COMPONENT_SOURCES
        test/model/LinearModel.cpp # Assuming this is source for the LinearModel test
        # Add other .cpp files for models or optimizers here if they exist
)
target_sources(ml_components PRIVATE ${ML_COMPONENT_SOURCES})
# Note: If ml_components has source files, you might want to make it a STATIC library.
# For this example, keeping it INTERFACE might be fine if sources are mostly tests.
# If it contains compiled model/optimizer code, change to add_library(ml_components STATIC ...)

# Link ml_components to autodiff as it likely depends on it
target_link_libraries(ml_components INTERFACE autodiff)


# --- Add example executables ---
add_executable(linear_neural_comparison src/examples/models/linear_neural_comparison.cpp)
target_link_libraries(linear_neural_comparison PRIVATE ml_components autodiff) # Link to components and core autodiff

add_executable(NeuralHiddenSizeComparison src/examples/models/NeuralHiddenSizeComparison.cpp)
target_link_libraries(NeuralHiddenSizeComparison PRIVATE ml_components autodiff) # Link to components and core autodiff

add_executable(newton_test src/examples/newton/test.cpp)
target_link_libraries(newton_test PRIVATE ml_components autodiff) # Link to components and core autodiff

# --- Add test executables ---
# The test/model files seem to be separate test entry points.
# Link these to the necessary libraries (ml_components and autodiff).

add_executable(linear_model_test test/model/LinearModel.cpp)
target_link_libraries(linear_model_test PRIVATE ml_components autodiff) # Link to ml_components (which includes LinearModel.cpp and links autodiff)

add_executable(neural_test test/model/NeuralTest.cpp)
target_link_libraries(neural_test PRIVATE ml_components autodiff) # Link to ml_components and autodiff

add_executable(optimizer_linear_test test/model/OptimizerLinearTest.cpp)
target_link_libraries(optimizer_linear_test PRIVATE ml_components autodiff) # Link to ml_components and autodiff

add_executable(span_test test/model/spantest.cpp)
target_link_libraries(span_test PRIVATE ml_components autodiff) # Link to ml_components and autodiff

add_executable(span_modelsize_test test/model/spanModelSizeTest.cpp)
target_link_libraries(span_modelsize_test PRIVATE ml_components autodiff)


# Linking executable files to OpenMP library for parallelization
if(OpenMP_FOUND)
    target_link_libraries(span_modelsize_test PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and linked for CXX.") # Optional: for confirmation
else()
    message(WARNING "OpenMP not found. Parallel features will be disabled or cause errors.")
endif()

# If src/autodiff/reverse/test.cpp is a separate test executable:
# add_executable(autodiff_reverse_test src/autodiff/reverse/test.cpp)
# target_link_libraries(autodiff_reverse_test PRIVATE autodiff)

# Note: CLion will automatically handle creating the cmake-build-debug directory
# and running CMake when you open the project.