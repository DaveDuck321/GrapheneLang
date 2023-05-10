@require_once "std/arithmetic.c3";

function main : () -> int = {
    let stack_array : i32[2, 2, 2];
    let heap_array : i32[&, 2, 2] = &stack_array;

    heap_array[0, 0, 1] = 1;
    heap_array[0, 1, 1] = 1;
    heap_array[1, 0, 1] = 1;
    heap_array[1, 1, 1] = 1;

    if (heap_array[0, 0, 1] != 1) {
        return 1;
    }
    if (heap_array[0, 1, 1] != 1) {
        return 1;
    }
    if (heap_array[1, 0, 1] != 1) {
        return 1;
    }
    if (heap_array[1, 1, 1] != 1) {
        return 1;
    }

    return 0;
};

/// @COMPILE
/// @GREP_IR getelementptr inbounds [2 x [2 x i32]]
/// @RUN
