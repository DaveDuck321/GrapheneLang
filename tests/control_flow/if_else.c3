@require_once "std/arithmetic.c3"

function main : () -> int = {
    let a : int = 0;

    if 7 == 8 {
        return 1;
    } else {
        a = 9;
    }

    if a == 9 {
        return 0;
    }
    return 1;
}

/// @COMPILE
/// @RUN
