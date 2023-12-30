function main : () -> int = {
    let a : f32 = 1e-38;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: 'f32' cannot represent '1e-38' since there is insufficient precision for a normalized representation.
