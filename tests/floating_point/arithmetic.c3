@require_once "std/arithmetic.c3"
@require_once "std/math.c3"

// TODO: bootstrap properly, move this into the standard library
function [T] abs : (value : T) -> TypeIf<T, IsFloating<T>> = {
    if value < 0.0 {
        return -value;
    } else {
        return value;
    }
}

function fp_are_equal : (a : f32, b : f32) -> bool = {
    return abs(a - b) < 1e-10;
}

function main : () -> int = {
    if !fp_are_equal(0.1 + 0.2, 0.3) {
        return 1;
    }

    if 0.29999 >= 0.3 {
        return 2;
    }

    if 0.30001 <= 0.3 {
        return 3;
    }

    if 0.5 + 0.5 != 1.0 {
        return 4;
    }

    return 0;
}

/// @COMPILE
/// @RUN
