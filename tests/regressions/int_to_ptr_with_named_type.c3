@require_once "std/arithmetic.c3"

function main : () -> int = {
    // Called with a `GenericIntType`.
    let x : int& = &__builtin_int_to_ptr<int&>(100);

    // Called with a `NamedType` (a specialization of `CommonArithmeticType`).
    let y : int& = &__builtin_int_to_ptr<int&>(100 + 4);

    return 0;
}

/// @COMPILE
