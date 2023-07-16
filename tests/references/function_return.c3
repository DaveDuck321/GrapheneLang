@require_once "std/arithmetic.c3"

function test_fn : (arg : int&) -> int& = {
    return &arg;
}

function main : () -> int = {
    let a : int = 0;
    let a_ref : int& = &test_fn(&a);

    a_ref = 10;

    let b : int = 0;
    test_fn(&b) = 9;

    return a + b;
}

/// @COMPILE
/// @RUN; EXPECT 19
