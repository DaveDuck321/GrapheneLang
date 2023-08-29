typedef [T] Span : {}

function [T] foo : ()
    -> Span = {}

function main : () -> int = {
    foo<int>();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4, in 'function return type'
/// Error: generic 'Span' expects 1 argument but received 0
