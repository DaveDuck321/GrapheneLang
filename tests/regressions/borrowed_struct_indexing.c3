function main : () -> int = {
    let a : {a : int, b : int, c : int} = {1, 2, 3};
    return (&a).c;
}

/// @COMPILE
/// @RUN; EXPECT 3
