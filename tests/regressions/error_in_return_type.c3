typedef [T] Span : {}

// Hmm, the error is a bit misleading...
// The overload resolution fails due to SFINAE not parameter count.. but this is
// also kinda the point of SFINAE, if we fixed this `EnableIf` would break.
function [T] foo : ()
    -> Span = {}

function main : () -> int = {
    foo<int>();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 10, in 'function main: () -> int'
/// Error: overload resolution for function call 'foo<int>()' failed
/// Available overloads:
/// - function [[]T[]] foo<T> : () -> Span
