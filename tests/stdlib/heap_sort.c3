@require_once "std/algorithm.c3";
@require_once "std/arithmetic.c3";

function main : () -> int = {
    let array    : int[7];
    let expected : int[7];

    let i : int = 0;
    while (7 > i) {
        array[i] = 7 - i;
        expected[i] = i;

        i = i + 1;
    }

    // heap_sort<int>(array, 7);
    max_heapify<int>(&array, 7, 0);

    i = 0;
    while (7 > i) {
        if (array[i] != expected[i]) {
            return array[i];
        }

        i = i + 1;
    }

    return 0;
};

/// @COMPILE
/// @RUN
