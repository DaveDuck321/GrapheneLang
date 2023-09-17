function main : () -> int = {
    let a : int[3] = {1, 2, 3};
    let b : int[mut&] = &a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error in variable assignment: type 'int[[]3[]]&' does not match expected type 'int[[]mut&[]]'
