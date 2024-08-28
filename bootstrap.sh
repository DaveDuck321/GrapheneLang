#!/usr/bin/env bash

set -ex

function checkout() {
    # Pass commit in $1. Output directory is returned.
    rpm_dir="GrapheneLang-${1/\//-}"
    if [ -d "$rpm_dir" ]; then
        realpath "$rpm_dir"
    else
        dir="$(mktemp -d)"
        git archive "$1" --worktree-attributes ":(exclude)tests/" | tar -x -C "$dir"
        realpath "$dir"
    fi
}

mkdir -p ./dist

dest_dir=$PWD
stage_1_dir="$(checkout bootstrap/1)"
stage_2_dir="$(checkout bootstrap/2)"
stage_4_dir="$(checkout bootstrap/4)"
stage_5_dir="$(checkout bootstrap/5)"
stage_6_dir="$(mktemp -d)"

# Temporary work around to bootstrap the new array syntax
# This should be updated to us an actual tag once this PR is merged
cp -a "$dest_dir/." "$stage_6_dir/"

# We can't have these inside the function, as the EXIT signal seems to be raised
# when the functions returns... It also looks like a second call to trap
# overwrites the first one.
# shellcheck disable=SC2064
trap "rm -rf $stage_1_dir; rm -rf $stage_2_dir; rm -rf $stage_4_dir; rm -rf $stage_5_dir; rm -rf $stage_6_dir" EXIT INT QUIT TERM

# Stage 1; last commit where the Lark parser can parse the native parser.
"$stage_1_dir/glang" "$stage_1_dir/parser/parser.c3" -o "$stage_2_dir/parser/parser" --bootstrap

# Stage 2; last commit before the parser output format changes
"$stage_2_dir/glang" "$stage_2_dir/parser/parser.c3" -o "$stage_2_dir/parser/parser"

# Stage 3; the parser output format has changed, compile with the latest
# available compiler. The location of the output has also changed.
mkdir -p "$stage_4_dir/dist"
"$stage_2_dir/glang" "$stage_4_dir/src/glang/parser/parser.c3" -o "$stage_4_dir/dist/parser"

# Stage 4; we introduce `mut` syntax, first compile a commit where the parser
# (but not the codegen) supports it. note: from this point forward we run as a
# python module
mkdir -p "$stage_5_dir/dist"
env -C "$stage_4_dir" python3 -m src.glang.driver ./src/glang/parser/parser.c3 -o "$stage_5_dir/dist/parser"

# Stage 5; we introduce floating point literal support, compile the last parser
# that supports this before literals become required by the standard library.
env -C "$stage_5_dir" python3 -m src.glang.driver ./src/glang/parser/parser.c3 -o "$stage_5_dir/dist/parser"

# Stage 6; adds parser support for index operator overloading and function types
# This is especially awkward since array indexing is no-longer builtin:
# 1) The stage 5 parser cannot parse the syntax required to overload the array operator
# 2) The stage 6 parser requires that the index operator is overloaded with this same syntax
# These conflicting requirements are achived by deleting the file that defines these overloads
# so that the stage 5 compiler falls back to its builtins but a fresh checkout of the stage 6
# compiler will work as expected.
echo "// Bootstrap hack" > "$stage_6_dir/src/glang/lib/std/array.c3"
env -C "$stage_5_dir" python3 -m src.glang.driver "$stage_6_dir/src/glang/parser/parser.c3" --nostdlib -I "$stage_6_dir/src/glang/lib/" "$stage_6_dir/src/glang/lib/std/$(env -C "$stage_5_dir" python3 -m src.glang.driver --print-host-target)/" -o "$dest_dir/dist/parser"

# Final Stage; build the native parser from the working tree. Note that we need
# to invoke the driver as a module.
PYTHONPATH="$dest_dir/src:$PYTHONPATH" python3 -m src.glang.driver ./src/glang/parser/parser.c3 -o ./dist/parser -O3
