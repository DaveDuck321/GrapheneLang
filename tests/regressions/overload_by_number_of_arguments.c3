@require_once "std/io.c3"

function [T] fn : () -> void = {
    puts<6>("No arg");
}

function [T] fn : (arg : int&) -> void = {
    puts<3>("arg");
}

function main : () -> int = {
    let a : int = 0;
    fn<int>();
    fn<int>(&a);
    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// No arg
/// arg
