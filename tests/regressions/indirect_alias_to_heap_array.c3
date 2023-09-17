typedef [T] Identity : T
typedef Test : Identity<int[&]>

function main : () -> int = {
    mut b : int[1] = {1};
    let a : Test = &b;
    return 0;
}

/// @COMPILE
/// @RUN
