function main : () -> int = {
    let array : int[4] = { 0, 0, 1 };

    return array[3];
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: initializer list with 3 objects cannot be converted to an array with 4 members
