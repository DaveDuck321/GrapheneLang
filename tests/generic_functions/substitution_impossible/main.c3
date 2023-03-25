typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : int;

function[T] generic_fn : () -> EnableIf<Typo<T>> = {
    return 0;
};

function main : () -> int = {
    generic_fn<i8>();
    return 0;
};
