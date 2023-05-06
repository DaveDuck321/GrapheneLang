typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

typedef[T] IsI32 : FalseType;
typedef IsI32<i32> : TrueType;


function main : () -> int = {
    let a : EnableIf<IsI32<i8>> = 0;
    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 10, in 'function main: () -> int'
/// Error: no definition exists for Type 'EnableIf<IsI32<i8>>', it may be incorrectly specialized
