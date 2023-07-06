#!/bin/bash

# Get the directory of this bash script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Remove all the files in the build directory
rm -rf $DIR/build

if [ $? -eq 0 ]; then
    echo "Cleaned build directory"
else
    echo "Failed to clean build directory"
fi