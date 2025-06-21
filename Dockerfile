# Stage 1: Build Stage
# Use a base image with a modern C++ compiler. Ubuntu 22.04 is a good choice.
FROM ubuntu:22.04 AS build

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools, Git, and other dependencies required to get the latest CMake.
# The default CMake in Ubuntu 22.04 is too old, so we'll get the latest from Kitware's repository.
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    g++ \
    wget \
    gpg \
    ca-certificates && \
    # Add the Kitware repository to get an up-to-date version of CMake (required version >= 3.28)
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

# Install project-specific dependencies: Eigen3 and OpenMP
RUN apt-get install -y \
    libeigen3-dev \
    libomp-dev && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project source code into the working directory
COPY . .

# Create a build directory, run CMake to configure the project, and then build it.
# This compiles all the executables defined in the CMakeLists.txt file.
RUN cmake -B build -S . && \
    cmake --build build -j"$(nproc)"

# Stage 2: Final Stage
# Use a smaller, "distroless"-style base image for the final image to reduce size and attack surface.
# We only need to copy the compiled executables and their runtime dependencies.
FROM ubuntu:22.04

# We need to install the OpenMP runtime library, as it's a dynamic dependency for the executables.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*


# Set the working directory for the final image
WORKDIR /app

# Copy all the compiled executables from the build stage into the final image
COPY --from=build /app/build/linear_neural_comparison .
COPY --from=build /app/build/NeuralHiddenSizeComparison .
COPY --from=build /app/build/linear_model_test .
COPY --from=build /app/build/neural_test .
COPY --from=build /app/build/optimizer_linear_test .
COPY --from=build /app/build/span_test .
COPY --from=build /app/build/span_modelsize_test .
COPY --from=build /app/build/newton_test .
COPY --from=build /app/build/autodiff_reverse_test .

# Set the default command to run one of the test executables.
# You can override this when you run the container.
# Example: docker run <image_name> ./neural_test
CMD ["./linear_neural_comparison"]

