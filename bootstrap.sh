#!/usr/bin/env bash

set -ex

function checkout() {
    # Pass commit in $1 and output directory in $2.
    mkdir -p "$2"
    git archive --worktree-attributes "$1" | tar -x -C "$2"
}

# Stage 1; last commit where the Lark parser can parse the native parser.
stage_1_dir=./bootstrap-1

checkout bootstrap/1 $stage_1_dir
$stage_1_dir/glang $stage_1_dir/parser/parser.c3 -o ./parser/parser --bootstrap -O3

rm -rf $stage_1_dir

# Stage 2; build the native parser from the working tree using the binary from
# Stage 1.
./glang ./parser/parser.c3 -o ./parser/parser -O3
