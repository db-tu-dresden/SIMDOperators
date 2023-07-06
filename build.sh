#!/bin/bash

# Get the directory of this bash script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


CMAKE_DEFINITIONS=""
TARGET="HelloTSL"
ENABLE_TESTING=""

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -h,  --help            Display this message"
            echo "  -c,  --clean           Clean the build directory before build"
            echo "  -r,  --run             Run the project"
            echo "  -a,  --all             Clean, build, and run the project"
            echo "  -t,  --target <name>   Specify the target to build"
            echo "  -rt, --runtests        Build and run the tests"
            echo "       --sse             Enable SSE"
            echo "       --avx2            Enable AVX2"
            echo "       --avx512          Enable AVX512"
            echo "       --neon            Enable NEON"
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -r|--run)
            RUN=true
            shift
            ;;
        -a|--all)
            CLEAN=true
            RUN=true
            shift
            ;;
        -t|--target)
            TARGET="$2"
            shift
            shift
            ;;
        -rt|--runttests)
            BUILD=true
            RUN_TESTS=true
            ENABLE_TESTING="-DENABLE_TESTING=True"
            shift
            ;;
        --sse)
            CMAKE_DEFINITIONS="$CMAKE_DEFINITIONS -DUSE_SSE"
            shift
            ;;
        --avx2)
            CMAKE_DEFINITIONS="$CMAKE_DEFINITIONS -DUSE_AVX2"
            shift
            ;;
        --avx512)
            CMAKE_DEFINITIONS="$CMAKE_DEFINITIONS -DUSE_AVX512"
            shift
            ;;
        --neon)
            CMAKE_DEFINITIONS="$CMAKE_DEFINITIONS -DUSE_NEON"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Clean the build directory
if [ "$CLEAN" = true ]; then
    $DIR/clean.sh
fi


# CMake configuration
mkdir -p $DIR/build
cd $DIR/build
cmake -G Ninja $ENABLE_TESTING -DCMAKE_DEFINITIONS="$CMAKE_DEFINITIONS" ..
if [ $? -eq 0 ]; then
    echo "CMake configuration succeeded"
else
    echo "CMake configuration failed"
    exit 1
fi

# CMake build
echo "-----------------------BUILD-----------------------"
cmake --build . --target $TARGET
if [ $? -eq 0 ]; then
    echo "CMake build succeeded"
else
    echo "CMake build failed"
    exit 1
fi
EXECUTABLE=$(find . -type f -name $TARGET)
echo "Executable: $EXECUTABLE"

# Run tests
if [ "$RUN_TESTS" = true ]; then
    cd $DIR/build
    # build the tests
    cmake --build . --target AllTests
    # find the executable
    EXECUTABLE=$(find . -type f -name AllTests)
    if [ $? -eq 0 ]; then
        echo "-----------------------TESTS-----------------------"
        $EXECUTABLE
    else
        echo "Failed to find executable: AllTests"
        exit 1
    fi
fi

# Run the project
if [ "$RUN" = true ]; then
    cd $DIR
    # find the executable
    EXECUTABLE=$(find build -type f -name $TARGET)
    if [ $? -eq 0 ]; then
        echo "-----------------------RUN-----------------------"
        $EXECUTABLE
    else
        echo "Failed to find executable: $TARGET"
        exit 1
    fi

fi