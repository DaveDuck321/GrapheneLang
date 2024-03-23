@require_once "std/arithmetic.c3"
@require_once "std/iterators.c3"
@require_once "std/format.c3"

function is_less_than_20 : (arg: int) -> bool = {
    return arg < 20;
}

function main : () -> int = {
    mut i : int = 0;
    mut sum : int = 0;

    while is_less_than_20(i) {
        if (i == 10) {
            sum += 2 * i;
            break;
        }

        sum += i;
        i += 1;
    }

    for j in range(21) {
        if (j == 10) {
            sum += 2 * j;
            break;
        } else {
            sum += j;
            // Here we rely on the branch at the end of the loop.
        }
    }

    if sum == 130 {
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN
