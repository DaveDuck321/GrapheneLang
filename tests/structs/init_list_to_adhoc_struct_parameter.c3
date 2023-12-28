function test : (param : {a : int, b : int}) -> int = {
    return param.a;
}

function main : () -> int = {
    return test({1, 7});
}

/// @COMPILE
/// @RUN; EXPECT 1
