@require_once "std/arithmetic.c3"
@require_once "std/iterators.c3"

function main : () -> int = {
    let sum : int = 0;
    for i in range(10) {
        sum += i;
    }
    if sum != 45 {
        return 1;
    }

    sum = 0;
    for i in range(2, 10) {
        sum += i;
    }
    if sum != 44 {
        return 2;
    }

    sum = 0;
    for i in range(1, 10, 2) {
        sum += i;
    }
    if sum != 25 {
        return 3;
    }

    return 0;
}

/// @COMPILE
/// @RUN
