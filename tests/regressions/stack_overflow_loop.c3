@require_once "std/arithmetic.c3"

function main : () -> int = {
    mut i: int = 0;
    while i < 1000000 {
        // This causes a stack overflow if new space for `a` is allocated on
        //   every loop iteration
        let a : int = i;
        i += 1;
    }
    return 0;
}

/// @COMPILE
/// @RUN
