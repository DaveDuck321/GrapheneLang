function [T1, T2] test : (param : {a : T1, b : isize, c : T2}) -> T1 = {
    return param.a;
}

function main : () -> int = {
    return test({.a = 100, .b = 8, .c={1}});
}

/// @COMPILE
/// @RUN; EXPECT 100
