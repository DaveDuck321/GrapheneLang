function main: () -> i32 = {
    if {1, 2} {
        return 1;
    }
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> i32'
/// Error: initializer list of the form '{Literal<1>, Literal<2>}' cannot be converted to 'bool'
