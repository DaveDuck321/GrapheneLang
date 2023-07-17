function[Ts...] inner: (args: Ts...) -> int = {
    return 0;
}

function[T, Ys...] outer: (curr: T, next: Ys...) -> int = {
    return inner(undefined...);
}

function main : () -> int = {
    return outer(1, 2, 3);
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 6, in 'function outer<int, int, int>: (int, int, int) -> int'
/// Error: could not find parameter pack definition 'undefined...'
