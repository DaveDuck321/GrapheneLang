function main : () -> int = {
    let val : int;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// let val : int;
/// ^
// Error: invalid syntax, a non-mutable value cannot be declared uninitialized
