typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

typedef[T] IsI32 : FalseType;
typedef IsI32<i32> : TrueType;


function main : () -> int = {
    let a : EnableIf<IsI32<i8>> = 0;
    return 0;
};
