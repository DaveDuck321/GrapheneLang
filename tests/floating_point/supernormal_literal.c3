function main : () -> int = {
    let a : f32 = 3.402824e+38;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: 'f32' cannot represent '3.402824e+38' since it is too large and would be truncated to +infty. Only values below 3.402823e+38 can be represented.
