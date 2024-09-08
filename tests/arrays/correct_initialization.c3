@require_once "std/array.c3"

function main : () -> int = {
    let array : int[4] = { 0, 0, 1, 2 };

    return array[3];
}

/// @COMPILE
/// @RUN; EXPECT 2
