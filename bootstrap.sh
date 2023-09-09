#!/usr/bin/env bash

set -ex

function checkout() {
    # Pass commit in $1. Output directory is returned.
    rpm_dir="GrapheneLang-${1/\//-}"
    if [ -d "$rpm_dir" ]; then
        echo "$rpm_dir"
    else
        dir="$(mktemp -d)"
        git archive "$1" --worktree-attributes ":(exclude)tests/" | tar -x -C "$dir"
        echo "$dir"
    fi
}

mkdir -p ./dist

stage_1_dir="$(checkout bootstrap/1)"
stage_2_dir="$(checkout bootstrap/2)"

# We can't have these inside the function, as the EXIT signal seems to be raised
# when the functions returns... It also looks like a second call to trap
# overwrites the first one.
# shellcheck disable=SC2064
trap "rm -rf $stage_1_dir; rm -rf $stage_2_dir" EXIT INT QUIT TERM

# Stage 1; last commit where the Lark parser can parse the native parser.
"$stage_1_dir/glang" "$stage_1_dir/parser/parser.c3" -o "$stage_2_dir/parser/parser" --bootstrap

# Stage 2; last commit before the parser output format changes
"$stage_2_dir/glang" "$stage_2_dir/parser/parser.c3" -o "$stage_2_dir/parser/parser"

# Stage 3; the parser output format has changed, compile with the latest
# available compiler. The location of the output has also changed.
"$stage_2_dir/glang" ./src/glang/parser/parser.c3 -o ./dist/parser

# Final Stage; build the native parser from the working tree. Note that we need
# to invoke the driver as a module.
/usr/bin/env python3 -m src.glang.driver ./src/glang/parser/parser.c3 -o ./dist/parser -O3
