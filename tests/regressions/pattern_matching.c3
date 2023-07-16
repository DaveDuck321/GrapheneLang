function [T] foo : (a : T, b : isize) -> void = {

}

function main : () -> int = {
    foo(10, 10);
    return 0;
}

/// @COMPILE
/// @RUN
