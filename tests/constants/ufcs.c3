@require_once "std/arithmetic.c3"

typedef A : {}

function foo : (a : A&) -> int = {
    return 0;
}

function bar : (a : A&) -> int = {
    return 1;
}

function bar : (a : A mut&) -> int = {
    return 0;
}

function main : () -> int = {
    mut a : A = {};
    // Can match with A&, but should prefer A mut&.
    return a:foo() + a:bar();
}

/// @COMPILE
/// @RUN
