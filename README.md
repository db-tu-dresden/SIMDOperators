# SIMDOperators

## Get started
1. Check out the repository with all its submodules:

`git clone --recurse-submodules git@github.com:db-tu-dresden/SIMDOperators.git .`

2. (optional) Checkout all submodules to the main branch (otherwise you will be on a detached head)

`git submodule foreach --recursive git checkout main`

3. Generate the TSL into libs/tsl (or execute `tools/bash/generate.sh`):

`python3 tools/tslgen/main.py --targets sse sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 --no-workaround-warnings -o ../../libs/tsl`
Change the targets as you like :). If you need a C++17 Version add `--no-concepts` to the call

4. Build the sources (or execute `tools/bash/build.sh`)

`rm -rf build && mkdir build && cd build && cmake .. && make && cd -`
