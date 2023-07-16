@require_once "std/arithmetic.c3"

function is_less_than_20 : (arg: int) -> bool = {
    return __builtin_is_less_than(arg, 20);
}

function main : () -> int = {
    let i : int = 0;
    let sum : int = 0;

    while (is_less_than_20(i)) {
        i = i + 1;
        sum = sum + i;
    }
    return sum;
}


/// @COMPILE
/// @RUN; EXPECT 210
