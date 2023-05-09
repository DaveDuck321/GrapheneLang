typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

typedef Not<TrueType> : FalseType;
typedef Not<FalseType> : TrueType;

typedef[T] IsLargeIntegral : FalseType;
typedef IsLargeIntegral<i32> : TrueType;
typedef IsLargeIntegral<i64> : TrueType;
typedef IsLargeIntegral<i128> : TrueType;

foreign puts : (str : u8[&]) -> int;

function[T] generic_fn : (a : T) -> EnableIf<IsLargeIntegral<T>> = {
    puts("Large integral type\0");
    return 0;
};

function[T] generic_fn : (a : T) -> EnableIf<Not<IsLargeIntegral<T>>> = {
    puts("Small integral type\0");
    return 0;
};

function main : () -> int = {
    let small : i8 = __builtin_narrow<i8>(0);
    let big : i64 = 0;

    generic_fn(small);
    generic_fn(big);
    return 0;
};

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// Small integral type
/// Large integral type
