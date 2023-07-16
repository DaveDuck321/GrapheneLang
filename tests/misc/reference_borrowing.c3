function func : (var : int&) -> i32 = {
    return var;
}

function main: () -> int = {
    let a: int = 7;
    let b: int& = &a;
    return func(&b);
}

/// @COMPILE
/// @RUN; EXPECT 7
