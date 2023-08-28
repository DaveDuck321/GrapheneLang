#!/usr/bin/env bash

set -ex

# Attempt to get the current location. Prefer a branch name if available,
# otherwise fall back to the current commit's hash if we are in "detached HEAD"
# state.
git_location=$(git branch --show)
if [[ "$git_location" == "" ]]; then
    git_location=$(git rev-parse HEAD)
fi

# Stage 1; last commit where the Lark parser can parse the native parser.
git checkout bootstrap/1
./glang ./parser/parser.c3 -o ./parser/parser --bootstrap -O3

# Stage 2; build native parser at current commit
git checkout "$git_location"
./glang ./parser/parser.c3 -o ./parser/parser -O3
