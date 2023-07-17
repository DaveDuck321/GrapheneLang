@require_once "std/arithmetic.c3"

function is_less_than_20 : (arg: int) -> bool = {
    return __builtin_is_less_than(arg, 20);
}

function main : () -> int = {
    let i : int = 0;
    let sum : int = 0;

    while is_less_than_20(i) {
        i = i + 1;

        let a : int = 0;
        while 10 > a {
            a = a + 1;
            sum = sum + i;
        }
    }

    if sum != 2100{
        return 1;
    }
    return 0;
}


/// @COMPILE
/// @RUN
