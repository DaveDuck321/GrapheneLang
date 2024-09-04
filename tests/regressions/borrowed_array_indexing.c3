@require_once "std/array.c3"

function main : () -> int = {
    let a : int[3] = {1, 2, 3};
    return (&a)[2];
}

/// @COMPILE
/// @RUN; EXPECT 3
