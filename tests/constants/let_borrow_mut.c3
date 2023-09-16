function main : () -> int = {
    let val : int = 0;
    (&mut val);
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: cannot perform mutable borrow on non-mutable type 'int&'
