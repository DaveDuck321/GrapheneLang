function [@Len] hello : (a : u8[@Len]&) -> void = {}
function [T] hello : (a : T) -> void = {}

function main : () -> int = {
    hello(8);
    return 0;
}

/// @COMPILE
/// @RUN
