function main : () -> int = {
    let a : f32 = 3.4028236e+38;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: 'f32' cannot represent '3.4028236e+38' since it is too large and would be truncated to +infty. Only values below 3.4028235e+38 can be represented.
