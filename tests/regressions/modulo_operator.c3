@require_once "std/arithmetic.c3"

function main : () -> int = {
    return 3 % 2;
}

/// @COMPILE
/// @RUN; EXPECT 1
