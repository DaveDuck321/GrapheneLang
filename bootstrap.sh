#!/usr/bin/env bash

set -ex

function checkout() {
    # Pass commit in $1 and output directory in $2.
    mkdir -p "$2"
    git archive --worktree-attributes "$1" | tar -x -C "$2"
}

stage_1_dir=./bootstrap-1
stage_2_dir=./bootstrap-2

checkout bootstrap/1 $stage_1_dir
checkout bootstrap/2 $stage_2_dir

# Stage 1; last commit where the Lark parser can parse the native parser.
$stage_1_dir/glang $stage_1_dir/parser/parser.c3 -o $stage_2_dir/parser/parser --bootstrap

# Stage 2; last commit before the parser output format changes
$stage_2_dir/glang $stage_2_dir/parser/parser.c3 -o $stage_2_dir/parser/parser

# Stage 3; the parser output format has changed, compile with the latest available compiler
$stage_2_dir/glang ./src/glang/parser/parser.c3 -o ./src/glang/parser/parser

rm -rf $stage_1_dir
rm -rf $stage_2_dir

# Final Stage; build the native parser from the working tree
glang ./src/glang/parser/parser.c3 -o ./src/glang/parser/parser -O3
