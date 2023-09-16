function main: () -> int = {
    mut x: int;
    return x;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: cannot use uninitialized variable 'x'
