@require_once "std/type_traits.c3"
@require_once "std/logical.c3"

typedef A : {}
typedef [T] B : {}
typedef B<int> : A

function [T] value : () -> bool = {
    let thing : T = {};
    return value(thing);
}

function value : (t : TrueType) -> bool = {
    return true;
}

function value : (t : FalseType) -> bool = {
    return false;
}

function main : () -> int = {
    if value<AreEquivalent<A, B<isize>>>() {
        return 1;
    }
    if not value<AreEquivalent<A, B<int>>>() {
        return 2;
    }
    return 0;
}

/// @COMPILE
/// @RUN
