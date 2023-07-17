function main: () -> i32 = {
    {1, 2} = 7;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> i32'
/// Error: cannot assign to an initializer list
