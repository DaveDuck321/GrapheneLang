@require_once "std/io.c3"

function normal_overloaded_function : (a : i64) -> int = {
    puts("i64");
    return 0;
}

function normal_overloaded_function : (a : i32) -> int = {
    puts("i32");
    return 0;
}

function normal_overloaded_function : (a : i128) -> int = {
    puts("i128");
    return 0;
}

function[T] generic_fn : () -> int = {
    let a : T = 0;
    return normal_overloaded_function(a);
}

function main : () -> int = {
    generic_fn<i32>();
    generic_fn<i32>();
    generic_fn<i32>();
    generic_fn<i128>();
    return generic_fn<i64>();
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// i32
/// i32
/// i32
/// i128
/// i64
