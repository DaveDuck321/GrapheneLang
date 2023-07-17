function main : () -> int = {
    const array : int[4] = { 0, 0, 1, "hello!" };

    return array[3];
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error in array initialization: type 'u8[[]6[]]&' does not match expected type 'int'
