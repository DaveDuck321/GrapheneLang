typedef TrueType : {}
typedef FalseType : {}
typedef EnableIf<TrueType> : int

typedef isI32<i32> : TrueType

function[T] generic_fn : () -> EnableIf<isI32<T>> = {
    return 0;
}

function[T] generic_fn : () -> int = {
    return 0;
}


function main : () -> int = {
    generic_fn<i32>();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 17, in 'function main: () -> int'
/// Error: function call 'generic_fn()' is ambiguous. Equally good candidates are:
/// - function generic_fn<T> : () -> EnableIf<isI32<T>>      (File '*.c3', line 7)
/// - function generic_fn<T> : () -> int                     (File '*.c3', line 11)
