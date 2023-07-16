@require_once "std/arithmetic.c3"

function[T] foo: (i: isize, j: T) -> void = {}

function main : () -> int = {
    const i: isize = 2;

    // CommonArithmeticType<isize, int> should match against isize.
    foo(i + 1, 0);

    return 0;
}

/// @COMPILE
