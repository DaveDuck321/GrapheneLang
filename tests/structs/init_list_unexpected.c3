function main: () -> i32 = {
    if {1, 2} {
        return 1;
    }
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> i32'
/// Error: initializer list of the form '{int, int}' cannot be converted to 'bool'
