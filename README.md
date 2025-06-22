# C++ Automatic Differentiation Library

## Prerequisites

* A C++20 compliant compiler
* CMake (>=3.28)
* The Eigen library (>=3.4.0)
* Cuda (optional)

## Building

1.  Clone the repository
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the project and generate build files:
    ```bash
    cmake ..
    ```
    To enable CUDA computation, pass the following option to CMake:
    ```bash
    cmake -DENABLE_CUDA=ON ..
    ```
4.  Build the project
    ```bash
    make
    ```

This will compile the library, examples, and tests.

## Running Examples

Navigate to the build directory and run the examples:

```bash
cd build
./linear_neural_comparison
./NeuralHiddenSizeComparison
./newton_test
# ... and any other example executables you have
```

## Option using Docker
However, if you don't have all the dependencies already installed, and you want to create a virtual env separately, you can check the following:
If you are using Linux, you must activate the Docker daemon first, then run_interactive.sh will help to build a Docker environment. 
1. Activation of the daemon service
```
sudo systemctl activate docker
```
2. Make sure you are in the root folder of this project, autodiff
```
chmod u+x run_interactive.sh
./run_interactive
```
3. Then you can take a break. After some minutes, you will find yourself inside the Docker container!
   You can either run the tests or create the build folder in the project folder and start developing.

## Plotting results
Some tests generate CSV files; for some, there are corresponding Python scripts to plot the CSV results. You can call those scripts like the following:
```
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas
python plot_models.py
deactivate
```
You can create a Python environment to execute those files. This will make your installation clean.
