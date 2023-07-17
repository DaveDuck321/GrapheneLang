function main : () -> int = {
    const array : int[4] = { 0, 0, 1, 2 };

    return array[3];
}

/// @COMPILE
/// @RUN; EXPECT 2
