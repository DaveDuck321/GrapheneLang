@require_once "std/arithmetic.c3"


function main : () -> int = {
    let a : int = 9;
    let b : i8 = __builtin_narrow<i8>(7);

    a += b;
    a /= 2;
    a *= 9;
    a -= 1;
    return a;
}

/// @COMPILE
/// @RUN; EXPECT 71
