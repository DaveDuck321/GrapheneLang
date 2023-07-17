@require_once "std/arithmetic.c3"
@require_once "std/logical.c3"
@require_once "std/util.c3"

function main : () -> int = {
    let a : int = 1;
    if -a != -1 {
        return 1;
    }

    if +a != 1 {
        return 1;
    }

    let b : u32 = 0xFFFF0000;
    if b ^ 0xFFFFFFFF != 0xFFFF {
        return 1;
    }
    if ~(b ^ 0xFFFFFFFF) != 0xFFFF0000 {
        return 1;
    }

    return - a + - + - a;
}

/// @COMPILE
/// @RUN
