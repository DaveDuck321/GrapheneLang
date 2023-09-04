typedef TrueType : {}
typedef FalseType : {}
typedef EnableIf<TrueType> : int

function[T] generic_fn : () -> EnableIf<Typo<T>> = {
    return 0;
}

function main : () -> int = {
    generic_fn<i8>();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 10, in 'function main: () -> int'
/// Error: could not find type definition 'typedef Typo<i8> : ...'
