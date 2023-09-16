function main : () -> int = {
    let a : int = 0;
    a = 1;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: left hand side of assignment operation has non-mutable type 'int&'
