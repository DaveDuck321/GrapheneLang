function [T] hello : () -> void = {}

function main : () -> int = {
    hello();
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4, in 'function main: () -> int'
/// Error: cannot deduce generic type 'T' in function 'hello', manual specialization is required
