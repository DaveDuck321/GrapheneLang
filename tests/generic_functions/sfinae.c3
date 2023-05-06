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

function[T] generic_fn : () -> EnableIf<IsLargeIntegral<T>> = {
    puts("Large integral type");
    return 0;
};

function[T] generic_fn : () -> EnableIf<Not<IsLargeIntegral<T>>> = {
    puts("Small integral type");
    return 0;
};

function main : () -> int = {
    generic_fn<i8>();
    generic_fn<i32>();
    return 0;
};

/// @COMPILE
/// @RUN; EXPECT OUT
/// Small integral type
/// Large integral type
