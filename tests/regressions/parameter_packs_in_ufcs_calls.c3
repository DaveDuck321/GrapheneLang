@require_once "std/io.c3"

typedef UFCS : {}

function fn : (_ : UFCS&) -> void = {
    puts("This should never be called");
}

function [Args...] fn : (_ : UFCS&, args : Args...) -> void = {
    puts("This should be called");
}

function [Args...] launcher : (args : Args...) -> void = {
    let ufcs : UFCS = {};
    ufcs:fn(args...);
}

function main : () -> int = {
    launcher(9, 8);
    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// This should be called
