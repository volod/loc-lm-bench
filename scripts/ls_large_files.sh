#!/bin/bash
# List the K largest files in the current git repository.
set -e

TOP_K=${1:-10}

git ls-tree -r --long HEAD | sort -k 4 -n -r | head -n $TOP_K | awk '{printf "%-10s %s\n", $4, $5}'

