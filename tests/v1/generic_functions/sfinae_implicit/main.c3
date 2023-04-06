@require_once "std/io.c3";

typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

typedef Not<TrueType> : FalseType;
typedef Not<FalseType> : TrueType;

typedef[T] IsLargeIntegral : FalseType;
typedef IsLargeIntegral<i32> : TrueType;
typedef IsLargeIntegral<i64> : TrueType;
typedef IsLargeIntegral<i128> : TrueType;

function[T] generic_fn : (a : T) -> EnableIf<IsLargeIntegral<T>> = {
    puts("Large integral type");
    return 0;
};

function[T] generic_fn : (a : T) -> EnableIf<Not<IsLargeIntegral<T>>> = {
    puts("Small integral type");
    return 0;
};

function main : () -> int = {
    let small : i8 = __builtin_narrow<i8>(0);
    let big : i64 = 0;

    generic_fn(small);
    generic_fn(big);
    return 0;
};
