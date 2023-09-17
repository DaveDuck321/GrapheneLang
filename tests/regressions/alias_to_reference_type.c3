typedef [T] Identity : T

function [T] fn : (iter : T) -> Identity<T> = {
    return &iter;
}

function main : () -> int = {
    let data : int = 0;
    return fn(&data);
}

/// @COMPILE
/// @RUN
