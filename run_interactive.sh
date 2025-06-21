#!/bin/bash

# This script checks if the Docker image for the autodiff project exists.
# If not, it builds it. Then, it launches an interactive bash shell
# inside a container.

# Define the name for the docker image
IMAGE_NAME="autodiff-app"

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Check if Docker image exists and build if necessary ---
# The '-q' flag makes 'docker images' output only the IMAGE ID.
# If the image doesn't exist, the output will be an empty string.
if [ -z "$(docker images -q $IMAGE_NAME 2> /dev/null)" ]; then
    echo "Image '$IMAGE_NAME' not found."
    echo "Building the Docker image..."
    docker build -t $IMAGE_NAME .
    echo "Docker image built successfully."
else
    echo "Docker image '$IMAGE_NAME' already exists. Using existing image."
fi

# --- Run the Container Interactively ---
echo "---"
echo "Launching an interactive shell in the container..."
echo "You can run any of the compiled executables, for example: ./newton_test"
echo "Type 'exit' to leave the container."
echo "---"

# --rm      : Automatically remove the container when it exits.
# -it       : Allocate a pseudo-TTY and keep STDIN open (interactive mode).
# $IMAGE_NAME : The image to run.
# /bin/bash : Overrides the default CMD and starts a bash shell.
docker run --rm -it $IMAGE_NAME /bin/bash

