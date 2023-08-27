function main: () -> i32 = {
    {1, 2} = 7;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// {1, 2} = 7;
/// ^
/// Error: invalid syntax, illegal assignment operator
