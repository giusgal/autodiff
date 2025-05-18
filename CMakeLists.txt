cmake_minimum_required(VERSION 3.31)
project(autodiff)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spantest src/examples/models/spantest.cpp)
add_executable(neuraltest src/examples/models/NeuralTest.cpp)
add_executable(optimizerlineartest test/model/OptimizerLinearTest.cpp)

