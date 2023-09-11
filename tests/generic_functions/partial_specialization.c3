@require_once "std/arithmetic.c3"
@require_once "std/logical.c3"
@require_once "std/type_traits.c3"

function [Ret, Val] test_fn : (a : Val) -> TypeIf<Ret, AreEquivalent<Ret, bool>> = {
    return false;
}

function [Ret, Val] test_fn : (a : Val) -> TypeIf<Ret, AreEquivalent<Ret, int>> = {
    return 1;
}

function main : () -> int = {
    if test_fn<bool>(7) {
        return 1;
    }

    if test_fn<int>(false) != 1 {
        return 2;
    }
    return 0;
}

/// @COMPILE
/// @RUN
