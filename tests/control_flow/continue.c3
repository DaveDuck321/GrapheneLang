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
        if (i == 9) {
            i += 1;
            sum += 2 * i;
            continue;
        } else {
            i += 1;
            sum += i;
            // Here we need to suppress the branch at the end of the loop.
            continue;
        }
    }

    for j in range(21) {
        if (j == 10) {
            sum += 2 * j;
            continue;
        } else {
            sum += j;
            // Here we rely on the branch at the end of the loop.
        }
    }

    if sum == 440 {
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN
