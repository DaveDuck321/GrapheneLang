@require_once "std/arithmetic.c3"

function main : () -> int = {
    0x7 >> 1;
    0x6 << 1;
    return 3 % 2;
}

/// @COMPILE
/// @RUN; EXPECT 1
