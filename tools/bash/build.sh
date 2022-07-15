#!/bin/bash
CPATH=$( dirname -- "$0"; )
if [ "$CPATH" = "." ]; then
	CPATH="../.."
else
	CPATH="."
fi

rm -rf $CPATH/build && mkdir $CPATH/build && cd $CPATH/build && cmake .. && make && cd -
