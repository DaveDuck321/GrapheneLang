typedef TrueType : {}
typedef FalseType : {}

typedef Not<TrueType> : FalseType
typedef Not<FalseType> : TrueType

typedef EnableIf<TrueType> : {}

typedef [T] Thing1 : EnableIf<T>
typedef [T] Thing1 : EnableIf<Not<T>>

typedef [T] Thing2 : EnableIf<Not<T>>
typedef [T] Thing2 : EnableIf<T>

function main : () -> int = {
    let a : Thing1<TrueType> = {};
    let b : Thing2<TrueType> = {};
    return 0;
}

/// @COMPILE
/// @RUN
