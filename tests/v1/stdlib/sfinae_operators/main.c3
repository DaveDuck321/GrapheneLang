@require_once "std/arithmetic.c3";
@require_once "std/type_traits.c3";

typedef isI128<i128> : TrueType;

function[T] assertIsI128 : (arg: T) -> TypeIf<i128, isI128<T>> = {
    return arg;
};

function main : () -> int = {
    let a : i8 = __builtin_narrow<i8>(1);
    let b : i16 = __builtin_narrow<i16>(2);
    let c : i32 = 4;
    let d : i64 = 8;
    let e : i128 = 16;

    let sum : i128 = assertIsI128(a + b + d + e + c);
    return __builtin_narrow<i32>(sum);
};
