@require_once "std/arithmetic.c3"

function test_fn : (arg : int mut&) -> int mut& = {
    return &mut arg;
}

function main : () -> int = {
    mut a : int = 0;
    let a_ref : int mut& = &mut test_fn(&mut a);

    a_ref = 10;

    mut b : int = 0;
    test_fn(&mut b) = 9;

    return a + b;
}

/// @COMPILE
/// @RUN; EXPECT 19
