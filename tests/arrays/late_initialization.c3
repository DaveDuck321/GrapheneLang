@require_once "std/array.c3"

function main : () -> int = {
    mut array : int[4];

    array = { 0, 0, 1, 2 };

    return array[3];
}

/// @COMPILE
/// @RUN; EXPECT 2
