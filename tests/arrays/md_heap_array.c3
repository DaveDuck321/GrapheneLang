@require_once "std/arithmetic.c3"

function main : () -> int = {
    mut stack_array : i32[2, 2, 2];
    let heap_array : i32[mut&, 2, 2] = &mut stack_array;

    heap_array[0, 0, 1] = 1;
    heap_array[0, 1, 1] = 1;
    heap_array[1, 0, 1] = 1;
    heap_array[1, 1, 1] = 1;

    if heap_array[0, 0, 1] != 1 {
        return 1;
    }
    if heap_array[0, 1, 1] != 1 {
        return 1;
    }
    if heap_array[1, 0, 1] != 1 {
        return 1;
    }
    if heap_array[1, 1, 1] != 1 {
        return 1;
    }

    return 0;
}

/// @FAILING
/// @COMPILE
/// @GREP_IR getelementptr inbounds [[]2 x [[]2 x i32[]][]]
/// @RUN
