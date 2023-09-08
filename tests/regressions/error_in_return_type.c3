typedef [T] Span : {}

// TODO: the line number still isn't right here
function [T] foo : ()
    -> Span = {}

function main : () -> int = {
    foo<int>();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 8, in 'function main: () -> int'
/// Error: generic type 'Span' received 0 arguments. This conflicts with the following definitions that expect 1 argument:
/// - typedef  [[]T[]] Span<T> : {}      (File '*.c3', line 1)
