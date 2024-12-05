#!/bin/bash
CPATH=$( dirname -- "$0"; )
if [ "$CPATH" = "." ]; then
	CPATH="../.."
else
	CPATH="."
fi

git submodule foreach --recursive git checkout main
python3 $CPATH/tools/tslgen/main.py --targets sse sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 --no-workaround-warnings -o ../../libs/tsl
