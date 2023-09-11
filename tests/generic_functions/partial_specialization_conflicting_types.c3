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
    if test_fn<bool, bool>(7) {
        return 1;
    }
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 14, in 'function main: () -> int'
/// Error: overload resolution for function call 'test_fn<bool, bool>(int)' failed
/// Available overloads:
/// - function [[]Ret, Val[]] test_fn<Ret, Val> : (Val) -> TypeIf<Ret, AreEquivalent<Ret, bool>>
/// - function [[]Ret, Val[]] test_fn<Ret, Val> : (Val) -> TypeIf<Ret, AreEquivalent<Ret, int>>
