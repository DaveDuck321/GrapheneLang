function [T] a : (a : T) -> isize = {
    return a;
}

function main : () -> int = {
    a<isize>(8);
    return 0;
}

/// @COMPILE
/// @RUN
