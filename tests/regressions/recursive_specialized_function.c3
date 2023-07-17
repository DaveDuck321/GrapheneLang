@require_once "std/arithmetic.c3"

function [T] recursive: (i: int) -> int = {
    if i > 0{
        recursive<T>(i - 1);
    }

    return 0;
}

function main : () -> int = {
    return recursive<int>(2);
}

/// @COMPILE
/// @RUN; EXPECT 0
