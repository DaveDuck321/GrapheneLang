@require_once "std/algorithms.c3"
@require_once "std/arithmetic.c3"

function main : () -> int = {
    let array_1 : int[5] = {0, 2, 4, 6, 8};
    const result_1 : isize = bisect_left(array_1:make_span(), 5);

    if result_1 != 3 {
        return 1;
    }

    let array_2 : int[1] = {0};
    const result_2 : isize = bisect_left(array_2:make_span(), 5);

    if result_2 != 1 {
        return 1;
    }

    let array_3 : int[6] = {-2, 0, 2, 5, 5, 10};
    const result_3 : isize = bisect_left(array_3:make_span(), 5);

    if result_3 != 3 {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN
