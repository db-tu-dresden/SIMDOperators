#!/bin/bash

# Get the directory of this bash script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Remove all the files in the build directory
rm -rf $DIR/build

# Prompt if submodules should be updated
read -p "Update submodules? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git submodule update --init --recursive
else
    echo "Not updating submodules"
fi


if [ $? -eq 0 ]; then
    echo "Cleaned build directory"
else
    echo "Failed to clean build directory"
fi