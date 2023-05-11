function main : () -> int = {
    let array : int[4];

    array = { 0, 0, 1, 2 };

    return array[3];
};

/// @COMPILE
/// @RUN; EXPECT 2
