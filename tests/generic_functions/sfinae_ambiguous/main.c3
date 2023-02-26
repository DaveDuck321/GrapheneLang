typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

typedef isI32<i32> : TrueType;

function[T] generic_fn : () -> EnableIf<isI32<T>> = {
    return 0;
};

function[T] generic_fn : () -> int = {
    return 0;
};


function main : () -> int = {
    generic_fn<i32>();
    return 0;
};
