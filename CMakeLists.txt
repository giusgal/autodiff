cmake_minimum_required(VERSION 3.28) # Minimum required CMake version

project(autodiff_project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# OpenMp dependency
find_package(OpenMP REQUIRED)
# Eigen dependency
find_package(Eigen3 3.4 REQUIRED NO_MODULE)


# Option to enable CUDA support
option(ENABLE_CUDA "Enable CUDA compilation" OFF)

# Enable CUDA if requested
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found at: ${CUDAToolkit_ROOT}")
        message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "CUDA Toolkit not found!")
    endif()

    # --gpu-architecture sm_75
    set(CMAKE_CUDA_ARCHITECTURES "75") 
    
    # Add CUDA-specific compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    
    # Add preprocessor definitions for CUDA
    
    

    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support disabled")
endif()


# Define include directories for the entire project
include_directories(
        ${CMAKE_SOURCE_DIR}/include
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

if(ENABLE_CUDA)
    # Add CUDA include directories to autodiff library
    target_include_directories(autodiff INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    
    # Link CUDA runtime to autodiff library
    target_link_libraries(autodiff INTERFACE CUDA::cudart)
endif()

# --- Build a library for models and optimizers ---
# This groups headers and potentially sources for models and optimizers.
add_library(ml_components INTERFACE)

# Link their include directories
target_include_directories(ml_components INTERFACE
        ${CMAKE_SOURCE_DIR}/include/examples/ml/models
        ${CMAKE_SOURCE_DIR}/include/examples/ml/optimizer
)



# Add any source files for models/optimizers if they exist outside includes
set(ML_COMPONENT_SOURCES
        ${CMAKE_SOURCE_DIR}/src/examples/ml/models
)

target_sources(ml_components PRIVATE ${ML_COMPONENT_SOURCES})
# Note: If ml_components has source files, you might want to make it a STATIC library.
# For this example, keeping it INTERFACE might be fine if sources are mostly tests.
# If it contains compiled model/optimizer code, change to add_library(ml_components STATIC ...)

# Link ml_components to autodiff as it likely depends on it
target_link_libraries(ml_components INTERFACE autodiff Eigen3::Eigen)

# --- Build a library for newton solver ---
add_library(newton INTERFACE)

# Link their include directories
target_include_directories(newton INTERFACE
        ${CMAKE_SOURCE_DIR}/include/examples/newton
)

# Add any source files for newton if they exist outside includes
set(NEWTON_SOURCES
        ${CMAKE_SOURCE_DIR}/src/examples/newton
)

target_sources(newton PRIVATE ${NEWTON_SOURCES})

# Link newton to autodiff as it likely depends on it
target_link_libraries(newton INTERFACE autodiff Eigen3::Eigen)


# --- Add ML example executables ---
add_executable(linear_neural_comparison src/examples/ml/models/linear_neural_comparison.cpp)
target_link_libraries(linear_neural_comparison PRIVATE ml_components autodiff Eigen3::Eigen) # Link to components and core autodiff

add_executable(NeuralHiddenSizeComparison src/examples/ml/models/NeuralHiddenSizeComparison.cpp)
target_link_libraries(NeuralHiddenSizeComparison PRIVATE ml_components autodiff Eigen3::Eigen) # Link to components and core autodiff

add_executable(linear_model_test src/examples/ml/models/LinearModel.cpp)
target_link_libraries(linear_model_test PRIVATE ml_components autodiff Eigen3::Eigen) # Link to ml_components (which includes LinearModel.cpp and links autodiff)

add_executable(neural_test src/examples/ml/models/NeuralTest.cpp)
target_link_libraries(neural_test PRIVATE ml_components autodiff Eigen3::Eigen) # Link to ml_components and autodiff

add_executable(optimizer_linear_test src/examples/ml/models/OptimizerLinearTest.cpp)
target_link_libraries(optimizer_linear_test PRIVATE ml_components autodiff Eigen3::Eigen) # Link to ml_components and autodiff

add_executable(span_test src/examples/ml/models/spantest.cpp)
target_link_libraries(span_test PRIVATE ml_components autodiff Eigen3::Eigen) # Link to ml_components and autodiff

add_executable(span_modelsize_test src/examples/ml/models/spanModelSizeTest.cpp)
target_link_libraries(span_modelsize_test PRIVATE ml_components autodiff Eigen3::Eigen)


# --- Add Newton example executables ---
add_executable(newton_test src/examples/newton/test-newton.cpp)
target_link_libraries(newton_test PRIVATE newton autodiff Eigen3::Eigen) # Link to components and core autodif


# --- Add forward example executables --- 
add_executable(forward_jacobian_test src/autodiff/forward/test-jacobian.cpp)
target_link_libraries(forward_jacobian_test PRIVATE autodiff Eigen3::Eigen)

# Linking executable files to OpenMP library for parallelization
if(OpenMP_FOUND)
    target_link_libraries(span_modelsize_test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(span_test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(newton_test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(forward_jacobian_test PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and linked for CXX.") # Optional: for confirmation
else()
    message(WARNING "OpenMP not found. Parallel features will be disabled or cause errors.")
endif()

# Link CUDA libraries to autodiff if CUDA is enabled
if(ENABLE_CUDA)
    add_executable(test_cuda_jac src/autodiff/forward/test-cuda-jac.cu)
    target_link_libraries(test_cuda_jac PRIVATE autodiff Eigen3::Eigen CUDA::cudart CUDA::cublas)

    add_compile_definitions(USE_CUDA EIGEN_USE_GPU)

    set_property(TARGET test_cuda_jac PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

# --- Add reverse example executables --- TODO: remove 
# If src/autodiff/reverse/test.cpp is a separate test executable:
# add_executable(autodiff_reverse_test src/autodiff/reverse/test.cpp)
# target_link_libraries(autodiff_reverse_test PRIVATE autodiff Eigen3::Eigen)


enable_testing()

# --- Forward tests ---
add_executable(dualvar_test test/dualvar_test.cpp)
target_link_libraries(dualvar_test autodiff GTest::gtest_main)

add_executable(fw_diff_test test/forward_utility_test.cpp)
target_link_libraries(fw_diff_test autodiff GTest::gtest_main Eigen3::Eigen)

# --- Reverse tests ---
add_executable(var_test test/var_test.cpp)
target_link_libraries(var_test autodiff GTest::gtest_main)

add_executable(reverse_utility_test test/reverse_utility_test.cpp)
target_link_libraries(reverse_utility_test autodiff GTest::gtest_main Eigen3::Eigen)

# ArenaAllocator tests
add_executable(arena_allocator_test test/arena_allocator_test.cpp)
target_link_libraries(arena_allocator_test autodiff GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(dualvar_test)
gtest_discover_tests(fw_diff_test)
gtest_discover_tests(var_test)
gtest_discover_tests(reverse_utility_test)
gtest_discover_tests(arena_allocator_test)

# Note: CLion will automatically handle creating the cmake-build-debug directory
# and running CMake when you open the project.
